import argparse
import json
import math
import os
import sys
from typing import Dict, Iterable, List, Optional

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_scripts.jukebox_utils import load_jukebox_model
from processing.gpu_audio_augmentation import GPUAudioToMelSpectrogram
from metadata.title_key import (
    build_title_key_metadata_by_source,
    key_metadata_counts,
    key_metadata_for_path,
)
from windowed_data_utils import build_level_starts, build_timing_tensor, source_stem


WINDOWED_MANIFEST = 'windowed_manifest.jsonl'
SOURCE_METADATA = 'source_metadata.json'
DEFAULT_WINDOW_OVERLAP_FRACTION = 0.50
DEFAULT_AUDIO_EXTENSIONS = ('.wav', '.flac', '.aiff', '.aif')


def _require_metadata_path(metadata_path: Optional[str]) -> str:
    if not metadata_path:
        raise ValueError(
            'metadata_path is required for quantization preprocessing. '
            'Pass --metadata_path pointing to the MAESTRO CSV/JSON metadata file.'
        )
    resolved = os.path.abspath(os.path.expanduser(metadata_path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f'MAESTRO metadata file not found: {resolved}')
    return resolved


def _build_key_metadata_lookup(metadata_path: Optional[str], infer_missing_mode_as: str) -> Dict[str, dict]:
    metadata_path = _require_metadata_path(metadata_path)
    metadata_by_source = build_title_key_metadata_by_source(
        metadata_path,
        infer_missing_mode_as=infer_missing_mode_as,
    )
    if not metadata_by_source:
        raise ValueError(
            f'No title-key metadata entries could be built from {metadata_path}. '
            'Check that the file is a MAESTRO CSV/JSON with audio_filename or midi_filename columns.'
        )
    counts = key_metadata_counts(metadata_by_source)
    print(
        f'Title-key metadata: loaded {len(metadata_by_source)} lookup aliases from {metadata_path}; '
        f'key_counts={counts}'
    )
    return metadata_by_source


def _write_source_metadata(output_dir: str, source_metadata: Dict[str, dict]) -> None:
    if not source_metadata:
        return
    path = os.path.join(output_dir, SOURCE_METADATA)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(source_metadata, f, indent=2, sort_keys=True)


def _step_from_overlap(window_size: int, overlap_fraction: float, level_name: str) -> int:
    """!
    @brief Calculate the step size from the window size and overlap fraction.
    @param window_size The size of the window in frames for the level.
    @param overlap_fraction The desired overlap fraction between windows (e.g., 0.75 for 75% overlap).
    @param level_name The name of the level (e.g., 'top', 'middle', 'bottom') for error messages.
    @return The calculated step size in frames, ensuring it is at least 1 and does not exceed the window size.
    """
    
    if window_size <= 0:
        raise ValueError(f'{level_name}_time_frames must be > 0, got {window_size}')
    if not 0.0 <= overlap_fraction < 1.0:
        raise ValueError(f'overlap_fraction must be in [0, 1), got {overlap_fraction}')

    step = int(round(window_size * (1.0 - overlap_fraction)))
    return max(1, min(window_size, step))


def _resolve_step_frames(
    explicit_step: Optional[int],
    window_size: int,
    overlap_fraction: float,
    level_name: str,
) -> int:
    """!
    @brief Resolve the step frames for a given level, using either an explicit value or deriving from the overlap fraction.
    @param explicit_step An optional explicit step frame count. If provided, this value is used directly after validation.
    @param window_size The size of the window in frames for the level.
    @param overlap_fraction The desired overlap fraction between windows, used to derive the step if explicit_step is not provided.
    @param level_name The name of the level (e.g., 'top', 'middle', 'bottom') for error messages.
    @return The validated step size in frames for the requested level.
    """
    if explicit_step is not None:
        if explicit_step <= 0:
            raise ValueError(f'{level_name}_step_frames must be > 0, got {explicit_step}')
        return int(explicit_step)

    return _step_from_overlap(window_size, overlap_fraction, level_name)


def _encode_level_windows(model, batch_windows: np.ndarray, device: torch.device) -> torch.Tensor:
    """!
    @brief Encode a batch of spectrogram windows into VQ code indices.
    @param model Loaded Jukebox VQ-VAE model for one hierarchy level.
    @param batch_windows NumPy batch shaped (B, Freq, Time) containing normalized spectrogram windows.
    @param device Torch device used for the VQ-VAE forward pass.
    @return Integer tensor shaped (B, Time_latent, Freq_latent) on CPU for serialization.
    """
    x_batch = torch.from_numpy(batch_windows).unsqueeze(1).float().to(device)
    indices = model.encode_to_indices(x_batch)
    return indices.transpose(1, 2).contiguous().cpu().to(torch.int32)


def _list_audio_files(source_path: str, extensions: Iterable[str]) -> List[str]:
    extensions = tuple(str(ext).lower() for ext in extensions)
    files = []
    for root, _, names in os.walk(source_path):
        for name in names:
            if name.lower().endswith(extensions):
                files.append(os.path.join(root, name))
    return sorted(files)


def _build_audio_preprocessor(
    target_time_frames: int,
    sample_rate: int,
    frame_size: int,
    hop_length: int,
    n_mels: int,
    device: torch.device,
) -> GPUAudioToMelSpectrogram:
    preprocessor = GPUAudioToMelSpectrogram(
        sample_rate=sample_rate,
        target_time_frames=target_time_frames,
        n_fft=frame_size,
        hop_length=hop_length,
        n_mels=n_mels,
        random_downmix=False,
        pitch_shift_enabled=False,
    ).to(device)
    preprocessor.eval()
    return preprocessor


def _read_audio_window(
    file_path: str,
    start_frame: int,
    window_time_frames: int,
    target_sample_rate: int,
    hop_length: int,
) -> dict:
    info = sf.info(file_path)
    source_sample_rate = int(info.samplerate)
    target_num_samples = max(1, (int(window_time_frames) - 1) * int(hop_length))
    start_sample = int(round(int(start_frame) * int(hop_length) * source_sample_rate / target_sample_rate))
    read_frames = max(1, int(math.ceil(target_num_samples * source_sample_rate / target_sample_rate)))

    audio, _ = sf.read(
        file_path,
        start=max(0, start_sample),
        frames=read_frames,
        dtype='float32',
        always_2d=True,
    )
    if audio.shape[0] < read_frames:
        audio = np.pad(audio, ((0, read_frames - audio.shape[0]), (0, 0)), mode='constant')
    if audio.shape[1] == 1:
        audio = np.repeat(audio, repeats=2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]

    return {
        'waveform': torch.from_numpy(np.ascontiguousarray(audio.T)),
        'source_sample_rate': torch.tensor(source_sample_rate, dtype=torch.float32),
        'valid_samples': torch.tensor(audio.shape[0], dtype=torch.long),
        'path': file_path,
    }


def _collate_audio_window_dicts(items: List[dict]) -> dict:
    max_len = max(item['waveform'].shape[-1] for item in items)
    waveforms = []
    for item in items:
        waveform = item['waveform']
        if waveform.shape[-1] < max_len:
            waveform = torch.nn.functional.pad(waveform, (0, max_len - waveform.shape[-1]))
        waveforms.append(waveform)
    return {
        'waveform': torch.stack(waveforms, dim=0),
        'source_sample_rate': torch.stack([item['source_sample_rate'] for item in items]),
        'valid_samples': torch.stack([item['valid_samples'] for item in items]),
        'path': [item['path'] for item in items],
    }


def _audio_windows_to_specs(
    file_path: str,
    starts: List[int],
    window_time_frames: int,
    preprocessor: GPUAudioToMelSpectrogram,
    sample_rate: int,
    hop_length: int,
    device: torch.device,
) -> np.ndarray:
    items = [
        _read_audio_window(
            file_path=file_path,
            start_frame=start_frame,
            window_time_frames=window_time_frames,
            target_sample_rate=sample_rate,
            hop_length=hop_length,
        )
        for start_frame in starts
    ]
    batch = _collate_audio_window_dicts(items)
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    with torch.no_grad():
        specs = preprocessor(batch, augment=False)
    return specs.squeeze(1).detach().cpu().numpy().astype(np.float32, copy=False)


def _build_anchor_schedule(
    total_frames: int,
    top_time_frames: int,
    middle_time_frames: int,
    bottom_time_frames: int,
    top_step_frames: int,
    middle_step_frames: int,
    bottom_step_frames: int,
) -> Dict[int, List[str]]:
    """!
    @brief Build a schedule of anchor start frames mapped to their eligible 
    levels based on the provided window and step configurations.
    @param total_frames The total number of frames in the spectrogram.
    @param top_time_frames The window size in frames for the top level.
    @param middle_time_frames The window size in frames for the middle level.
    @param bottom_time_frames The window size in frames for the bottom level.
    @param top_step_frames The step size in frames for the top level.
    @param middle_step_frames The step size in frames for the middle level.
    @param bottom_step_frames The step size in frames for the bottom level.
    @return A dictionary mapping each anchor start frame (int) to a list of 
    eligible level names (List[str]) that can be trained at that anchor based
    on the window and step configurations.
    """
    top_starts = build_level_starts(total_frames, top_time_frames, top_step_frames)
    middle_starts = build_level_starts(total_frames, middle_time_frames, middle_step_frames)
    bottom_starts = build_level_starts(total_frames, bottom_time_frames, bottom_step_frames)

    anchor_to_levels: Dict[int, List[str]] = {}
    for level_name, starts in (
        ('top', top_starts),
        ('middle', middle_starts),
        ('bottom', bottom_starts),
    ):
        for start in starts:
            anchor_to_levels.setdefault(int(start), []).append(level_name)

    for levels in anchor_to_levels.values():
        levels.sort(key=lambda name: {'top': 0, 'middle': 1, 'bottom': 2}[name])
    return anchor_to_levels


def precompute_windowed_audio_examples(
    vqvae_dirs,
    source_audio_path,
    output_dir,
    top_time_frames=2048,
    middle_time_frames=512,
    bottom_time_frames=128,
    top_step_frames=None,
    middle_step_frames=None,
    bottom_step_frames=None,
    overlap_fraction=DEFAULT_WINDOW_OVERLAP_FRACTION,
    batch_size=8,
    sample_rate=22050,
    hop_length=256,
    frame_size=2048,
    n_mels=256,
    audio_extensions=DEFAULT_AUDIO_EXTENSIONS,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    weights_file='best_model.pth',
    metadata_path: Optional[str] = None,
    infer_missing_mode_as: str = 'major',
):
    """Quantize raw-audio files into windowed hierarchical VQ indices."""
    os.makedirs(output_dir, exist_ok=True)
    metadata_by_source = _build_key_metadata_lookup(metadata_path, infer_missing_mode_as)
    source_metadata = {}
    top_step_frames = _resolve_step_frames(top_step_frames, top_time_frames, overlap_fraction, 'top')
    middle_step_frames = _resolve_step_frames(middle_step_frames, middle_time_frames, overlap_fraction, 'middle')
    bottom_step_frames = _resolve_step_frames(bottom_step_frames, bottom_time_frames, overlap_fraction, 'bottom')
    effective_overlap = {
        'top': 1.0 - (top_step_frames / float(top_time_frames)),
        'middle': 1.0 - (middle_step_frames / float(middle_time_frames)),
        'bottom': 1.0 - (bottom_step_frames / float(bottom_time_frames)),
    }

    models = {
        lvl: load_jukebox_model(vqvae_dirs[lvl], lvl, device, weights_file).eval()
        for lvl in ['top', 'middle', 'bottom']
    }
    preprocessors = {
        'top': _build_audio_preprocessor(top_time_frames, sample_rate, frame_size, hop_length, n_mels, device),
        'middle': _build_audio_preprocessor(middle_time_frames, sample_rate, frame_size, hop_length, n_mels, device),
        'bottom': _build_audio_preprocessor(bottom_time_frames, sample_rate, frame_size, hop_length, n_mels, device),
    }

    files = _list_audio_files(source_audio_path, audio_extensions)
    if not files:
        raise FileNotFoundError(f'No audio files found in {source_audio_path} with extensions={audio_extensions}')

    manifest_path = os.path.join(output_dir, WINDOWED_MANIFEST)
    config_path = os.path.join(output_dir, 'windowed_quantization_config.json')
    config_payload = {
        'format': 'windowed_v1',
        'source_mode': 'audio',
        'source_path': source_audio_path,
        'top_time_frames': int(top_time_frames),
        'middle_time_frames': int(middle_time_frames),
        'bottom_time_frames': int(bottom_time_frames),
        'top_step_frames': int(top_step_frames),
        'middle_step_frames': int(middle_step_frames),
        'bottom_step_frames': int(bottom_step_frames),
        'requested_overlap_fraction': float(overlap_fraction),
        'effective_overlap_fraction': effective_overlap,
        'batch_size': int(batch_size),
        'sample_rate': int(sample_rate),
        'hop_length': int(hop_length),
        'frame_size': int(frame_size),
        'n_mels': int(n_mels),
        'audio_extensions': list(audio_extensions),
        'weights_file': weights_file,
        'vqvae_dirs': dict(vqvae_dirs),
        'metadata_path': metadata_path,
        'key_infer_missing_mode_as': infer_missing_mode_as,
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_payload, f, indent=2)

    total_examples = 0
    file_frame_counts = {}
    for file_path in files:
        info = sf.info(file_path)
        total_frames = max(1, int(math.floor(int(info.frames) * sample_rate / int(info.samplerate) / hop_length)) + 1)
        file_frame_counts[file_path] = total_frames
        total_examples += len(
            _build_anchor_schedule(
                total_frames=total_frames,
                top_time_frames=top_time_frames,
                middle_time_frames=middle_time_frames,
                bottom_time_frames=bottom_time_frames,
                top_step_frames=top_step_frames,
                middle_step_frames=middle_step_frames,
                bottom_step_frames=bottom_step_frames,
            )
        )

    print(
        'Precomputing raw-audio windowed hierarchical indices '
        f'(files={len(files)}, examples={total_examples}, top_step={top_step_frames}, '
        f'middle_step={middle_step_frames}, bottom_step={bottom_step_frames}, batch_size={batch_size})...'
    )

    with open(manifest_path, 'w', encoding='utf-8') as manifest_file, torch.no_grad():
        progress = tqdm(total=total_examples, desc='Quantizing Raw-Audio Windowed Examples')
        current_batch_size = max(1, int(batch_size))

        for file_path in files:
            total_frames = int(file_frame_counts[file_path])
            source_file_stem = source_stem(file_path)
            source_basename = os.path.basename(file_path)
            key_metadata = key_metadata_for_path(file_path, metadata_by_source)
            source_metadata[source_file_stem] = dict(key_metadata)
            anchor_to_levels = _build_anchor_schedule(
                total_frames=total_frames,
                top_time_frames=top_time_frames,
                middle_time_frames=middle_time_frames,
                bottom_time_frames=bottom_time_frames,
                top_step_frames=top_step_frames,
                middle_step_frames=middle_step_frames,
                bottom_step_frames=bottom_step_frames,
            )
            starts = sorted(anchor_to_levels.keys())

            i = 0
            while i < len(starts):
                end = min(i + current_batch_size, len(starts))
                batch_starts = starts[i:end]
                try:
                    top_batch = _audio_windows_to_specs(
                        file_path, batch_starts, top_time_frames, preprocessors['top'], sample_rate, hop_length, device
                    )
                    middle_batch = _audio_windows_to_specs(
                        file_path, batch_starts, middle_time_frames, preprocessors['middle'], sample_rate, hop_length, device
                    )
                    bottom_batch = _audio_windows_to_specs(
                        file_path, batch_starts, bottom_time_frames, preprocessors['bottom'], sample_rate, hop_length, device
                    )

                    top_indices = _encode_level_windows(models['top'], top_batch, device)
                    middle_indices = _encode_level_windows(models['middle'], middle_batch, device)
                    bottom_indices = _encode_level_windows(models['bottom'], bottom_batch, device)

                    for batch_idx, start_frame in enumerate(batch_starts):
                        eligible_levels = anchor_to_levels[int(start_frame)]
                        filename = f'{source_file_stem}__start_{int(start_frame):08d}_window_quantized.pt'
                        payload = {
                            'format': 'windowed_v1',
                            'source_mode': 'audio',
                            'source_basename': source_basename,
                            'source_stem': source_file_stem,
                            'start_frame': int(start_frame),
                            'total_frames': total_frames,
                            'timing': build_timing_tensor(
                                start_frame=int(start_frame),
                                total_frames=total_frames,
                                sample_rate=sample_rate,
                                hop_length=hop_length,
                            ),
                            'metadata': dict(key_metadata),
                            'eligible_levels': list(eligible_levels),
                            'top': top_indices[batch_idx].clone(),
                            'middle': middle_indices[batch_idx].clone(),
                            'bottom': bottom_indices[batch_idx].clone(),
                        }
                        torch.save(payload, os.path.join(output_dir, filename))
                        manifest_file.write(
                            json.dumps(
                                {
                                    'file': filename,
                                    'source_basename': source_basename,
                                    'source_stem': source_file_stem,
                                    'start_frame': int(start_frame),
                                    'total_frames': total_frames,
                                    'eligible_levels': list(eligible_levels),
                                    'key_id': int(key_metadata.get('key_id', 24)),
                                    'key_label': key_metadata.get('key_label', 'unknown'),
                                    'key_source': key_metadata.get('key_source', 'unknown'),
                                }
                            ) + '\n'
                        )

                    progress.update(end - i)
                    i = end
                except RuntimeError as exc:
                    if 'out of memory' in str(exc).lower() and device.type == 'cuda' and current_batch_size > 1:
                        new_batch_size = max(1, current_batch_size // 2)
                        print(
                            f'CUDA OOM while raw-audio quantizing {source_basename} at batch_size={current_batch_size}. '
                            f'Retrying with batch_size={new_batch_size}.'
                        )
                        current_batch_size = new_batch_size
                        torch.cuda.empty_cache()
                        continue
                    progress.close()
                    raise

            if device.type == 'cuda':
                torch.cuda.empty_cache()

        progress.close()

    _write_source_metadata(output_dir, source_metadata)


def main():
    """!
    @brief Parse CLI arguments and run windowed quantization preprocessing.
    The selected quantized dataset is written to disk and progress is printed to stdout.
    """
    parser = argparse.ArgumentParser(
        description='Preprocess raw audio and quantize it for hierarchical transformer prior training.'
    )
    
    parser.add_argument(
        '--source_path',
        type=str,
        default='./data/code/datasets/raw/maestro-v3.0.0/',
        help='Path to directory containing source raw audio files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/processed/backing_tracks_quantized_dataset/',
        help='Output directory for quantized .pt files'
    )
    parser.add_argument(
        '--top_model_dir',
        type=str,
        default='./models/jukebox_vq_vae/jukebox_vqvae_maestro_top/2026-05-14_10-01-43',
        help='Path to trained top-level VQ-VAE model'
    )
    parser.add_argument(
        '--middle_model_dir',
        type=str,
        default='./models/jukebox_vq_vae/jukebox_vqvae_maestro_middle/2026-05-15_16-56-29',
        help='Path to trained middle-level VQ-VAE model'
    )
    parser.add_argument(
        '--bottom_model_dir',
        type=str,
        default='./models/jukebox_vq_vae/jukebox_vqvae_maestro_bottom/2026-05-16_03-35-41',
        help='Path to trained bottom-level VQ-VAE model'
    )
    parser.add_argument(
        '--weights_file',
        type=str,
        default='best_model.pth',
        help='Checkpoint filename for VQ-VAE models'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for computation (cuda or cpu)'
    )
    parser.add_argument('--top_time_frames', type=int, default=2048, help='Top-level window (same as preprocess_audio TARGET_TIME_FRAMES)')
    parser.add_argument('--middle_time_frames', type=int, default=512, help='Middle-level window')
    parser.add_argument('--bottom_time_frames', type=int, default=128, help='Bottom-level window')
    parser.add_argument(
        '--overlap_fraction',
        type=float,
        default=DEFAULT_WINDOW_OVERLAP_FRACTION,
        help='Default overlap used to derive step frames when explicit step args are omitted (default: 0.50).',
    )
    parser.add_argument(
        '--top_step_frames',
        type=int,
        default=None,
        help='Top-level anchor step. Omit to derive from --overlap_fraction; with defaults this is 1024.',
    )
    parser.add_argument(
        '--middle_step_frames',
        type=int,
        default=None,
        help='Middle-level anchor step. Omit to derive from --overlap_fraction; with defaults this is 256.',
    )
    parser.add_argument(
        '--bottom_step_frames',
        type=int,
        default=None,
        help='Bottom-level anchor step. Omit to derive from --overlap_fraction; with defaults this is 64.',
    )
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for windowed quantization.')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Target sample rate used for raw-audio mel spectrograms')
    parser.add_argument('--hop_length', type=int, default=256, help='Hop length used for raw-audio mel spectrograms')
    parser.add_argument('--frame_size', type=int, default=2048, help='FFT/window size used for raw-audio mel spectrograms')
    parser.add_argument('--n_mels', type=int, default=256, help='Mel bin count used for raw-audio mel spectrograms')
    parser.add_argument(
        '--audio_extensions',
        nargs='+',
        default=list(DEFAULT_AUDIO_EXTENSIONS),
        help='Audio extensions to quantize.',
    )
    parser.add_argument(
        '--metadata_path',
        type=str,
        required=True,
        help='Required MAESTRO CSV/JSON metadata path used to infer title-derived key labels.',
    )
    parser.add_argument(
        '--key_infer_missing_mode_as',
        type=str,
        default='major',
        choices=['major', 'minor', 'unknown'],
        help='Mode assigned to title keys without an explicit major/minor marker, e.g. "Sonata Bb".',
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.isdir(args.source_path):
        print(f"Error: Source path does not exist: {args.source_path}")
        sys.exit(1)
    
    for model_path in [args.top_model_dir, args.middle_model_dir, args.bottom_model_dir]:
        if not os.path.isdir(model_path):
            print(f"Error: Model path does not exist: {model_path}")
            sys.exit(1)
    
    device = torch.device(args.device)
    top_step_frames = _resolve_step_frames(
        args.top_step_frames, args.top_time_frames, args.overlap_fraction, 'top'
    )
    middle_step_frames = _resolve_step_frames(
        args.middle_step_frames, args.middle_time_frames, args.overlap_fraction, 'middle'
    )
    bottom_step_frames = _resolve_step_frames(
        args.bottom_step_frames, args.bottom_time_frames, args.overlap_fraction, 'bottom'
    )
    effective_overlap = {
        'top': 1.0 - (top_step_frames / float(args.top_time_frames)),
        'middle': 1.0 - (middle_step_frames / float(args.middle_time_frames)),
        'bottom': 1.0 - (bottom_step_frames / float(args.bottom_time_frames)),
    }
    print(f'Using device: {device}')
    
    vqvae_dirs = {
        'top': args.top_model_dir,
        'middle': args.middle_model_dir,
        'bottom': args.bottom_model_dir,
    }
    
    print(f'Source path: {args.source_path}')
    print(f'Output directory: {args.output_dir}')
    print(f'Weights file: {args.weights_file}')
    print(
        f'Temporal config: top={args.top_time_frames}, middle={args.middle_time_frames}, '
        f'bottom={args.bottom_time_frames}, top_step={top_step_frames}, '
        f'middle_step={middle_step_frames}, bottom_step={bottom_step_frames}, '
        f'sr={args.sample_rate}, hop={args.hop_length}'
    )
    print(
        'Window overlap: '
        f"requested={args.overlap_fraction:.3f}, "
        f"effective top={effective_overlap['top']:.3f}, "
        f"middle={effective_overlap['middle']:.3f}, "
        f"bottom={effective_overlap['bottom']:.3f}"
    )
    print()

    precompute_windowed_audio_examples(
        vqvae_dirs=vqvae_dirs,
        source_audio_path=args.source_path,
        output_dir=args.output_dir,
        top_time_frames=args.top_time_frames,
        middle_time_frames=args.middle_time_frames,
        bottom_time_frames=args.bottom_time_frames,
        top_step_frames=top_step_frames,
        middle_step_frames=middle_step_frames,
        bottom_step_frames=bottom_step_frames,
        overlap_fraction=args.overlap_fraction,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        frame_size=args.frame_size,
        n_mels=args.n_mels,
        audio_extensions=args.audio_extensions,
        device=device,
        weights_file=args.weights_file,
        metadata_path=args.metadata_path,
        infer_missing_mode_as=args.key_infer_missing_mode_as,
    )
    
    print(f'\n✓ Quantization preprocessing complete!')
    print(f'Quantized files saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
