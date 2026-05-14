import math
from typing import Optional, Sequence

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio.functional as AF


class GPUAudioToMelSpectrogram(nn.Module):
    """
    Convert raw stereo audio batches to normalized log-mel spectrograms on GPU.

    The pitch augmentation uses the speed/resampling trick: it reads a slightly
    longer or shorter source span and resamples it to the fixed training length,
    avoiding phase-vocoder artifacts.
    """

    def __init__(
        self,
        sample_rate: int,
        target_time_frames: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        random_downmix: bool = True,
        downmix_weight_min: float = 0.0,
        downmix_weight_max: float = 1.0,
        pitch_shift_enabled: bool = True,
        min_pitch_shift_semitones: float = -0.5,
        max_pitch_shift_semitones: float = 0.5,
        pitch_shift_choices: Optional[Sequence[float]] = None,
        resample_lowpass_filter_width: int = 64,
        resample_chunk_size: int = 8192,
        max_torchaudio_resample_factor: int = 256,
        amin: float = 1e-10,
        top_db: Optional[float] = 80.0,
    ):
        super().__init__()
        if target_time_frames < 1:
            raise ValueError(f"target_time_frames must be >= 1, got {target_time_frames}")
        if n_fft < 1:
            raise ValueError(f"n_fft must be >= 1, got {n_fft}")
        if hop_length < 1:
            raise ValueError(f"hop_length must be >= 1, got {hop_length}")
        if n_mels < 1:
            raise ValueError(f"n_mels must be >= 1, got {n_mels}")
        if downmix_weight_min > downmix_weight_max:
            raise ValueError("downmix_weight_min must be <= downmix_weight_max")
        if min_pitch_shift_semitones > max_pitch_shift_semitones:
            raise ValueError("min_pitch_shift_semitones must be <= max_pitch_shift_semitones")
        if pitch_shift_choices is not None and len(pitch_shift_choices) == 0:
            raise ValueError("pitch_shift_choices must contain at least one value when provided")
        if resample_lowpass_filter_width < 1:
            raise ValueError("resample_lowpass_filter_width must be >= 1")
        if resample_chunk_size < 1:
            raise ValueError("resample_chunk_size must be >= 1")
        if max_torchaudio_resample_factor < 1:
            raise ValueError("max_torchaudio_resample_factor must be >= 1")

        self.sample_rate = int(sample_rate)
        self.target_time_frames = int(target_time_frames)
        self.target_num_samples = max(1, (self.target_time_frames - 1) * int(hop_length))
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.random_downmix = bool(random_downmix)
        self.downmix_weight_min = float(downmix_weight_min)
        self.downmix_weight_max = float(downmix_weight_max)
        self.pitch_shift_enabled = bool(pitch_shift_enabled)
        self.min_pitch_shift_semitones = float(min_pitch_shift_semitones)
        self.max_pitch_shift_semitones = float(max_pitch_shift_semitones)
        self.resample_lowpass_filter_width = int(resample_lowpass_filter_width)
        self.resample_chunk_size = int(resample_chunk_size)
        self.max_torchaudio_resample_factor = int(max_torchaudio_resample_factor)
        self.amin = float(amin)
        self.top_db = None if top_db is None else float(top_db)
        if pitch_shift_choices is None:
            self.pitch_shift_choices = None
        else:
            choices = torch.tensor([float(value) for value in pitch_shift_choices], dtype=torch.float32)
            self.register_buffer("pitch_shift_choices", choices, persistent=False)

        window = torch.hann_window(self.n_fft, periodic=True, dtype=torch.float32)
        self.register_buffer("window", window, persistent=False)

        mel_filter = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=0.0,
            fmax=float(self.sample_rate) / 2.0,
            htk=False,
            norm="slaney",
        ).astype(np.float32)
        self.register_buffer("mel_filter", torch.from_numpy(mel_filter), persistent=False)

    def forward(
        self,
        batch: dict,
        augment: bool = True,
        return_min_max: bool = False,
        return_waveform: bool = False,
        return_debug_info: bool = False,
    ):
        waveform = batch["waveform"].float()
        source_sample_rates = batch["source_sample_rate"].float()
        valid_samples = batch["valid_samples"].long()

        augment_output = self._augment_to_mono_waveforms(
            waveform=waveform,
            source_sample_rates=source_sample_rates,
            valid_samples=valid_samples,
            augment=augment,
            return_debug_info=return_debug_info,
        )
        if return_debug_info:
            mono, debug_infos = augment_output
        else:
            mono = augment_output
            debug_infos = None

        spec_db = self._log_mel_spectrogram(mono)
        normalized, min_vals, max_vals = self._normalize_per_sample(spec_db)
        normalized = normalized.unsqueeze(1)

        if not return_min_max and not return_waveform and not return_debug_info:
            return normalized

        outputs = [normalized]
        if return_min_max:
            min_max_values = [
                {"min": float(min_vals[i].item()), "max": float(max_vals[i].item())}
                for i in range(normalized.shape[0])
            ]
            outputs.append(min_max_values)
        if return_waveform:
            outputs.append(mono)
        if return_debug_info:
            outputs.append(debug_infos)
        return tuple(outputs)

    def _augment_to_mono_waveforms(
        self,
        waveform: torch.Tensor,
        source_sample_rates: torch.Tensor,
        valid_samples: torch.Tensor,
        augment: bool,
        return_debug_info: bool = False,
    ) -> torch.Tensor:
        batch_size = waveform.shape[0]
        output = []
        debug_infos = []

        for i in range(batch_size):
            valid = int(valid_samples[i].item())
            sample = waveform[i, :, :valid]

            if sample.shape[0] == 1:
                left = right = sample[0]
            else:
                left = sample[0]
                right = sample[1]

            if augment and self.random_downmix:
                weight = torch.empty((), device=waveform.device).uniform_(
                    self.downmix_weight_min,
                    self.downmix_weight_max,
                )
            else:
                weight = torch.tensor(0.5, device=waveform.device)
            mono = weight * left + (1.0 - weight) * right

            semitones = torch.tensor(0.0, device=waveform.device)
            if augment and self.pitch_shift_enabled:
                if self.pitch_shift_choices is not None:
                    choice_idx = torch.randint(0, self.pitch_shift_choices.numel(), (), device=waveform.device)
                    semitones = self.pitch_shift_choices[choice_idx]
                else:
                    semitones = torch.empty((), device=waveform.device).uniform_(
                        self.min_pitch_shift_semitones,
                        self.max_pitch_shift_semitones,
                    )
                speed = float(torch.pow(torch.tensor(2.0, device=waveform.device), semitones / 12.0).item())
            else:
                speed = 1.0

            source_sr = float(source_sample_rates[i].item())
            requested_source_span = int(math.ceil(self.target_num_samples * (source_sr / self.sample_rate) * speed))
            source_span = requested_source_span
            source_span = max(1, min(source_span, mono.shape[-1]))

            max_offset = max(0, mono.shape[-1] - source_span)
            if augment and max_offset > 0:
                offset = int(torch.randint(0, max_offset + 1, (), device=waveform.device).item())
            else:
                offset = max_offset // 2

            segment = mono[offset : offset + source_span]
            resample_output = self._resample_1d(
                segment,
                orig_freq=source_sr * speed,
                new_freq=float(self.sample_rate),
                target_num_samples=self.target_num_samples,
                return_resampler_name=return_debug_info,
            )
            if return_debug_info:
                segment, resampler_name = resample_output
            else:
                segment = resample_output
                resampler_name = None
            output.append(segment)

            if return_debug_info:
                debug_infos.append(
                    {
                        "sample_index": i,
                        "source_sample_rate": source_sr,
                        "target_sample_rate": float(self.sample_rate),
                        "downmix_weight_left": float(weight.detach().cpu().item()),
                        "downmix_weight_right": float((1.0 - weight).detach().cpu().item()),
                        "semitones": float(semitones.detach().cpu().item()),
                        "speed": speed,
                        "requested_source_span_samples": requested_source_span,
                        "source_span_samples": int(source_span),
                        "offset_samples": int(offset),
                        "output_samples": int(segment.shape[-1]),
                        "target_time_frames": int(self.target_time_frames),
                        "resampler": resampler_name,
                    }
                )

        stacked = torch.stack(output, dim=0)
        if return_debug_info:
            return stacked, debug_infos
        return stacked

    def _resample_1d(
        self,
        waveform: torch.Tensor,
        orig_freq: float,
        new_freq: float,
        target_num_samples: int,
        return_resampler_name: bool = False,
    ) -> torch.Tensor:
        if abs(orig_freq - new_freq) < 1e-6 and waveform.shape[-1] == target_num_samples:
            if return_resampler_name:
                return waveform, "none"
            return waveform

        if self._can_use_torchaudio_resample(orig_freq, new_freq):
            resampled = AF.resample(
                waveform,
                orig_freq=int(round(orig_freq)),
                new_freq=int(round(new_freq)),
                lowpass_filter_width=self.resample_lowpass_filter_width,
            )
            resampler_name = "torchaudio"
        else:
            resampled = self._sinc_resample_1d(
                waveform,
                orig_freq=orig_freq,
                new_freq=new_freq,
                target_num_samples=target_num_samples,
            )
            resampler_name = "torch_sinc"

        if resampled.shape[-1] > target_num_samples:
            resampled = resampled[:target_num_samples]
        elif resampled.shape[-1] < target_num_samples:
            resampled = F.pad(resampled, (0, target_num_samples - resampled.shape[-1]))

        if return_resampler_name:
            return resampled, resampler_name
        return resampled

    def _can_use_torchaudio_resample(self, orig_freq: float, new_freq: float) -> bool:
        if AF is None:
            return False
        orig_int = max(1, int(round(orig_freq)))
        new_int = max(1, int(round(new_freq)))
        gcd = math.gcd(orig_int, new_int)
        resample_factor = max(orig_int, new_int) // gcd
        return resample_factor <= self.max_torchaudio_resample_factor

    def _sinc_resample_1d(
        self,
        waveform: torch.Tensor,
        orig_freq: float,
        new_freq: float,
        target_num_samples: int,
    ) -> torch.Tensor:
        """Windowed-sinc resampling fallback for environments without torchaudio."""
        dtype = waveform.dtype
        device = waveform.device
        input_length = waveform.shape[-1]
        width = self.resample_lowpass_filter_width
        chunk_size = self.resample_chunk_size
        sample_rate_ratio = float(orig_freq) / float(new_freq)
        cutoff = min(1.0, float(new_freq) / float(orig_freq))
        tap_offsets = torch.arange(-width, width + 1, device=device, dtype=torch.float32)
        output_chunks = []

        for start in range(0, target_num_samples, chunk_size):
            end = min(start + chunk_size, target_num_samples)
            output_positions = torch.arange(start, end, device=device, dtype=torch.float32)
            input_positions = output_positions * sample_rate_ratio
            centers = torch.floor(input_positions).long()
            indices = centers[:, None] + tap_offsets.long()[None, :]
            distances = input_positions[:, None] - indices.to(torch.float32)

            window_mask = distances.abs() <= width
            window = 0.5 * (1.0 + torch.cos(math.pi * distances / float(width)))
            weights = cutoff * torch.sinc(cutoff * distances) * window * window_mask

            valid = (indices >= 0) & (indices < input_length)
            safe_indices = indices.clamp(0, max(0, input_length - 1))
            values = waveform[safe_indices] * valid.to(dtype)
            weights = weights.to(dtype)
            weight_sums = weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(dtype).eps)
            output_chunks.append((values * weights / weight_sums).sum(dim=1))

        return torch.cat(output_chunks, dim=0)

    def _log_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        power = stft.abs().pow(2.0)
        mel_power = torch.einsum("mf,bft->bmt", self.mel_filter, power)
        mel_power = torch.clamp(mel_power, min=self.amin)
        log_spec = 10.0 * torch.log10(mel_power)

        if self.top_db is not None:
            max_per_sample = log_spec.amax(dim=(1, 2), keepdim=True)
            log_spec = torch.maximum(log_spec, max_per_sample - self.top_db)

        return log_spec

    @staticmethod
    def _normalize_per_sample(spec: torch.Tensor):
        min_vals = spec.amin(dim=(1, 2), keepdim=True)
        max_vals = spec.amax(dim=(1, 2), keepdim=True)
        denom = torch.clamp(max_vals - min_vals, min=1e-12)
        normalized = (spec - min_vals) / denom
        return normalized, min_vals.flatten(), max_vals.flatten()


def _save_debug_plot(spec: np.ndarray, save_path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(spec, origin="lower", aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel bins")
    fig.colorbar(image, ax=ax, label="Normalized log-mel")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_debug_cli() -> None:
    import argparse
    import json
    import os
    import sys
    from pathlib import Path

    import soundfile as sf

    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if src_root not in sys.path:
        sys.path.append(src_root)

    from datasets.raw_audio_dataset import RawAudioWindowDataset, collate_audio_windows, list_audio_files

    parser = argparse.ArgumentParser(
        description="Inspect GPU audio augmentation by saving augmented WAVs, mel PNGs, and metadata."
    )
    parser.add_argument("--audio-path", type=str, default=None, help="Specific audio file to inspect.")
    parser.add_argument("--raw-dir", type=str, default="./data/raw/maestro-v3.0.0", help="Fallback directory when --audio-path is not provided.")
    parser.add_argument("--output-dir", type=str, default="./samples/gpu_audio_augmentation_debug", help="Directory for debug artifacts.")
    parser.add_argument("--num-examples", type=int, default=6, help="Number of random augmentations to render.")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--target-time-frames", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--n-mels", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-downmix", action="store_true", help="Disable random stereo downmix.")
    parser.add_argument("--downmix-weight-range", type=float, nargs=2, default=[0.0, 1.0])
    parser.add_argument("--no-pitch", action="store_true", help="Disable pitch augmentation.")
    parser.add_argument("--continuous-pitch", action="store_true", help="Use continuous semitone sampling instead of discrete choices.")
    parser.add_argument("--pitch-choices", type=float, nargs="*", default=[-2.0, -1.0, 0.0, 1.0, 2.0])
    parser.add_argument("--pitch-range", type=float, nargs=2, default=[-2.0, 2.0])
    parser.add_argument("--resample-lowpass-filter-width", type=int, default=64)
    parser.add_argument("--resample-chunk-size", type=int, default=8192)
    parser.add_argument("--max-torchaudio-resample-factor", type=int, default=256)
    args = parser.parse_args()

    if args.num_examples < 1:
        raise ValueError("--num-examples must be >= 1")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.audio_path is None:
        audio_paths = list_audio_files(args.raw_dir, extensions=[".wav"])
        if not audio_paths:
            raise FileNotFoundError(f"No .wav files found under {args.raw_dir}")
        audio_path = audio_paths[0]
    else:
        audio_path = args.audio_path

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pitch_enabled = not args.no_pitch
    pitch_choices = None if args.continuous_pitch or not pitch_enabled else args.pitch_choices
    pitch_range = args.pitch_range
    if pitch_choices:
        pitch_range = [min(pitch_choices), max(pitch_choices)]
    if not pitch_enabled:
        pitch_range = [0.0, 0.0]

    target_num_samples = max(1, (args.target_time_frames - 1) * args.hop_length)
    dataset = RawAudioWindowDataset(
        [audio_path],
        target_sample_rate=args.sample_rate,
        target_num_samples=target_num_samples,
        min_pitch_shift_semitones=float(pitch_range[0]),
        max_pitch_shift_semitones=float(pitch_range[1]),
        examples_per_file=args.num_examples,
        random_crop=True,
    )
    raw_batch = collate_audio_windows([dataset[i] for i in range(args.num_examples)])
    raw_batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in raw_batch.items()
    }

    augmenter = GPUAudioToMelSpectrogram(
        sample_rate=args.sample_rate,
        target_time_frames=args.target_time_frames,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        random_downmix=not args.no_downmix,
        downmix_weight_min=float(args.downmix_weight_range[0]),
        downmix_weight_max=float(args.downmix_weight_range[1]),
        pitch_shift_enabled=pitch_enabled,
        min_pitch_shift_semitones=float(pitch_range[0]),
        max_pitch_shift_semitones=float(pitch_range[1]),
        pitch_shift_choices=pitch_choices,
        resample_lowpass_filter_width=args.resample_lowpass_filter_width,
        resample_chunk_size=args.resample_chunk_size,
        max_torchaudio_resample_factor=args.max_torchaudio_resample_factor,
    ).to(device)

    with torch.no_grad():
        specs, min_max_values, augmented_waveforms, debug_infos = augmenter(
            raw_batch,
            augment=True,
            return_min_max=True,
            return_waveform=True,
            return_debug_info=True,
        )

    specs_np = specs.detach().cpu().numpy()
    waves_np = augmented_waveforms.detach().cpu().numpy()
    source_paths = raw_batch["path"]

    metadata = {
        "audio_path": audio_path,
        "device": str(device),
        "torchaudio_available": AF is not None,
        "sample_rate": args.sample_rate,
        "target_time_frames": args.target_time_frames,
        "target_num_samples": target_num_samples,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "n_mels": args.n_mels,
        "pitch_choices": pitch_choices,
        "pitch_range": pitch_range,
        "downmix_weight_range": args.downmix_weight_range,
        "samples": [],
    }

    for i, info in enumerate(debug_infos):
        wav_name = f"augmented_{i:03d}.wav"
        png_name = f"mel_{i:03d}.png"
        wav_path = output_dir / wav_name
        png_path = output_dir / png_name

        sf.write(wav_path, np.clip(waves_np[i], -1.0, 1.0), args.sample_rate)
        title = (
            f"shift={info['semitones']:+.2f} st, "
            f"L={info['downmix_weight_left']:.2f}, "
            f"speed={info['speed']:.4f}"
        )
        _save_debug_plot(specs_np[i, 0], png_path, title)

        sample_metadata = dict(info)
        sample_metadata.update(
            {
                "source_path": source_paths[i],
                "min_db": min_max_values[i]["min"],
                "max_db": min_max_values[i]["max"],
                "normalized_min": float(specs_np[i].min()),
                "normalized_max": float(specs_np[i].max()),
                "wave_path": wav_name,
                "mel_png_path": png_name,
            }
        )
        metadata["samples"].append(sample_metadata)

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {len(debug_infos)} augmented WAVs and mel plots to: {output_dir}")
    print(f"Metadata: {metadata_path}")
    print("First sample:")
    print(json.dumps(metadata["samples"][0], indent=2))


if __name__ == "__main__":
    _run_debug_cli()
