from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


DEFAULT_AUDIO_EXTENSIONS = (".wav", ".flac", ".aiff", ".aif")


def list_audio_files(root_dir: str, extensions: Optional[Iterable[str]] = None) -> list[str]:
    """Return sorted audio file paths under root_dir."""
    extensions = tuple(ext.lower() for ext in (extensions or DEFAULT_AUDIO_EXTENSIONS))
    paths = []
    for root, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.lower().endswith(extensions):
                paths.append(os.path.join(root, file_name))
    return sorted(paths)


class RawAudioWindowDataset(Dataset):
    """
    Lazily reads stereo waveform windows for GPU-side spectrogram augmentation.

    Each item is a dict so the training loop can keep source sample-rate metadata
    and perform the expensive resampling/STFT work on the target device.
    """

    def __init__(
        self,
        file_paths: Sequence[str],
        target_sample_rate: int,
        target_num_samples: int,
        min_pitch_shift_semitones: float = 0.0,
        max_pitch_shift_semitones: float = 0.0,
        examples_per_file: int = 1,
        random_crop: bool = True,
        crop_strategy: str = None,
    ):
        if target_num_samples < 1:
            raise ValueError(f"target_num_samples must be >= 1, got {target_num_samples}")
        if examples_per_file < 1:
            raise ValueError(f"examples_per_file must be >= 1, got {examples_per_file}")
        if crop_strategy is None:
            crop_strategy = "random" if random_crop else "center"
        crop_strategy = str(crop_strategy).lower()
        if crop_strategy not in ("random", "center", "non_overlapping"):
            raise ValueError("crop_strategy must be one of: random, center, non_overlapping")

        self.file_paths = [str(Path(p)) for p in file_paths]
        self.target_sample_rate = int(target_sample_rate)
        self.target_num_samples = int(target_num_samples)
        self.min_pitch_shift_semitones = float(min_pitch_shift_semitones)
        self.max_pitch_shift_semitones = float(max_pitch_shift_semitones)
        self.examples_per_file = int(examples_per_file)
        self.random_crop = bool(random_crop)
        self.crop_strategy = crop_strategy

        self._infos = []
        for path in self.file_paths:
            info = sf.info(path)
            self._infos.append(
                {
                    "path": path,
                    "sample_rate": int(info.samplerate),
                    "channels": int(info.channels),
                    "frames": int(info.frames),
                }
            )
        self._fixed_windows = self._build_fixed_windows() if self.crop_strategy == "non_overlapping" else None

    def __len__(self):
        if self._fixed_windows is not None:
            return len(self._fixed_windows)
        return len(self.file_paths) * self.examples_per_file

    def path_for_index(self, idx):
        if self._fixed_windows is not None:
            file_idx, _, _ = self._fixed_windows[idx]
            return self._infos[file_idx]["path"]
        file_idx = idx % len(self._infos)
        return self._infos[file_idx]["path"]

    def __getitem__(self, idx):
        if not self._infos:
            raise IndexError("RawAudioWindowDataset is empty.")

        if self._fixed_windows is not None:
            file_idx, start, read_frames = self._fixed_windows[idx]
            info = self._infos[file_idx]
        else:
            file_idx = idx % len(self._infos)
            info = self._infos[file_idx]
            read_frames = self._required_source_frames(info["sample_rate"])
            available_start = max(0, info["frames"] - read_frames)

            if self.crop_strategy == "random" and available_start > 0:
                start = int(np.random.randint(0, available_start + 1))
            else:
                start = max(0, available_start // 2)

        source_sample_rate = info["sample_rate"]

        audio, _ = sf.read(
            info["path"],
            start=start,
            frames=read_frames,
            dtype="float32",
            always_2d=True,
        )

        if audio.shape[0] < read_frames:
            pad = read_frames - audio.shape[0]
            audio = np.pad(audio, ((0, pad), (0, 0)), mode="constant")

        if audio.shape[1] == 1:
            audio = np.repeat(audio, repeats=2, axis=1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2]

        waveform = torch.from_numpy(np.ascontiguousarray(audio.T))
        return {
            "waveform": waveform,
            "source_sample_rate": torch.tensor(source_sample_rate, dtype=torch.float32),
            "valid_samples": torch.tensor(audio.shape[0], dtype=torch.long),
            "path": info["path"],
        }

    def _required_source_frames(self, source_sample_rate: int) -> int:
        max_speed = max(
            1.0,
            2.0 ** (self.min_pitch_shift_semitones / 12.0),
            2.0 ** (self.max_pitch_shift_semitones / 12.0),
        )
        sample_rate_ratio = float(source_sample_rate) / float(self.target_sample_rate)
        return max(1, int(math.ceil(self.target_num_samples * sample_rate_ratio * max_speed)))

    def _build_fixed_windows(self):
        windows = []
        for file_idx, info in enumerate(self._infos):
            read_frames = self._required_source_frames(info["sample_rate"])
            total_frames = max(0, int(info["frames"]))
            if total_frames <= read_frames:
                starts = [0]
            else:
                starts = list(range(0, total_frames, read_frames))
            for start in starts:
                windows.append((file_idx, int(start), int(read_frames)))
        return windows


def collate_audio_windows(batch: list[dict]) -> dict:
    """Pad variable-source-rate waveform windows so PyTorch can batch them."""
    if not batch:
        raise ValueError("Cannot collate an empty raw-audio batch.")

    max_len = max(item["waveform"].shape[-1] for item in batch)
    waveforms = []
    for item in batch:
        waveform = item["waveform"]
        if waveform.shape[-1] < max_len:
            waveform = F.pad(waveform, (0, max_len - waveform.shape[-1]))
        waveforms.append(waveform)

    return {
        "waveform": torch.stack(waveforms, dim=0),
        "source_sample_rate": torch.stack([item["source_sample_rate"] for item in batch]),
        "valid_samples": torch.stack([item["valid_samples"] for item in batch]),
        "path": [item["path"] for item in batch],
    }
