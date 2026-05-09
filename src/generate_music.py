import os
import pickle
import sys
import soundfile as sf
import argparse
import math
import json
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import numpy as np
import time

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_config, set_global_seed
from train_scripts.jukebox_utils import load_jukebox_model
from test_scripts.test_transformer_prior import load_transformer_prior
from generation.soundgenerator import SoundGenerator
from generation.transformer_io_utils import (
    decode_jukebox_indices,
    prepare_min_max_values,
    resolve_min_max_values_path,
    resolve_vqvae_config_path,
    save_level_spectrograms,
    save_decoded_spectrograms,
)
from windowed_data_utils import (
    build_level_starts,
    build_timing_schedule,
    build_windowed_starts,
)
from processing.preprocess_audio import SAMPLE_RATE, HOP_LENGTH, FRAME_SIZE



DEFAULT_TOP_RUN_ROOT = "models/transformer_prior/jukebox_maestro_top_transformer_prior"
DEFAULT_MIDDLE_RUN_ROOT = "models/transformer_prior/jukebox_maestro_middle_transformer_prior"
DEFAULT_BOTTOM_RUN_ROOT = "models/transformer_prior/jukebox_maestro_bottom_transformer_prior"


def _resolve_latest_config_path(run_root: str, level_name: str) -> str:
    if os.path.isfile(run_root):
        return run_root

    if not os.path.isdir(run_root):
        raise FileNotFoundError(f"{level_name} run root does not exist: {run_root}")

    direct_cfg = os.path.join(run_root, 'config.yaml')
    if os.path.isfile(direct_cfg):
        return direct_cfg

    candidates = []
    for entry in os.listdir(run_root):
        entry_path = os.path.join(run_root, entry)
        if not os.path.isdir(entry_path):
            continue
        cfg_path = os.path.join(entry_path, 'config.yaml')
        if os.path.isfile(cfg_path):
            candidates.append(cfg_path)

    if not candidates:
        raise FileNotFoundError(
            f"No config.yaml found in {run_root} for level {level_name}."
        )

    return max(candidates, key=os.path.getmtime)


def _resolve_prior_config_path(
    explicit_path: Optional[str],
    default_run_root: str,
    level_name: str,
) -> str:
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f"{level_name} config path not found: {explicit_path}")
        if os.path.isdir(explicit_path):
            return _resolve_latest_config_path(explicit_path, level_name)
        return explicit_path

    return _resolve_latest_config_path(default_run_root, level_name)


def _debug(msg: str) -> None:
    print(f'[DEBUG] {msg}')


def _generate_level_tokens(
    prior,
    seq_len: int,
    num_chunks: int,
    device: torch.device,
    temperature: float,
    top_k: Optional[int],
    upper_tokens_list: Optional[List[np.ndarray]] = None,
    second_upper_tokens_list: Optional[List[np.ndarray]] = None,
    timing_list: Optional[List[torch.Tensor]] = None,
    start_frames: Optional[List[int]] = None,
    level_time_frames: Optional[int] = None,
    level_grid: Optional[List[int]] = None,
    use_windowed_prefix: bool = False,
) -> List[np.ndarray]:
    token_blocks = []
    previous_tokens = None
    previous_start_frame = None

    for chunk in range(num_chunks):
        upper_indices = None
        if upper_tokens_list is not None:
            upper_indices = torch.from_numpy(upper_tokens_list[chunk]).to(device)

        second_upper_indices = None
        if second_upper_tokens_list is not None:
            second_upper_indices = torch.from_numpy(second_upper_tokens_list[chunk]).to(device)

        start_tokens = None
        if use_windowed_prefix and previous_tokens is not None:
            if start_frames is None or level_time_frames is None or level_grid is None:
                raise ValueError('start_frames, level_time_frames, and level_grid are required for windowed prefix sampling.')
            start_tokens_np = _extract_prefix_from_previous_window(
                previous_tokens=previous_tokens,
                previous_start_frame=previous_start_frame,
                current_start_frame=start_frames[chunk],
                level_time_frames=level_time_frames,
                level_grid=level_grid,
            )
            if start_tokens_np is not None and start_tokens_np.shape[1] > 0:
                start_tokens = torch.from_numpy(start_tokens_np).to(device)

        generate_kwargs = {
            'batch_size': 1,
            'start_tokens': start_tokens,
            'upper_indices': upper_indices,
            'second_upper_indices': second_upper_indices,
            'seq_len': seq_len,
            'temperature': temperature,
            'top_k': top_k,
            'device': device,
        }
        if timing_list is not None:
            timing = timing_list[chunk].to(device=device, dtype=torch.float32)
            generate_kwargs['timing'] = timing

        with torch.no_grad():
            tokens = prior.generate(**generate_kwargs).cpu().numpy()

        token_blocks.append(tokens)
        previous_tokens = tokens
        previous_start_frame = start_frames[chunk] if start_frames is not None else None

    return token_blocks


def _level_grid_info(level_time_frames: int, level_grid: List[int]) -> Tuple[int, int, int]:
    if not (isinstance(level_grid, (list, tuple)) and len(level_grid) == 2):
        raise ValueError(f'Expected level_grid=[time_cols, freq_bins], got {level_grid}')
    time_cols, freq_bins = int(level_grid[0]), int(level_grid[1])
    if time_cols <= 0 or freq_bins <= 0:
        raise ValueError(f'Invalid level grid: {level_grid}')
    if level_time_frames % time_cols != 0:
        raise ValueError(
            f'level_time_frames={level_time_frames} is not divisible by time_cols={time_cols}'
        )
    return time_cols, freq_bins, level_time_frames // time_cols


def _extract_prefix_from_previous_window(
    previous_tokens: np.ndarray,
    previous_start_frame: int,
    current_start_frame: int,
    level_time_frames: int,
    level_grid: List[int],
) -> Optional[np.ndarray]:
    """Use the overlapping tail of the previous window as autoregressive context."""
    time_cols, freq_bins, frames_per_token_col = _level_grid_info(level_time_frames, level_grid)
    delta_frames = int(current_start_frame) - int(previous_start_frame)
    if delta_frames <= 0:
        return None

    delta_cols = int(round(delta_frames / frames_per_token_col))
    if delta_cols >= time_cols:
        return None
    delta_cols = max(1, delta_cols)

    block = previous_tokens.reshape(previous_tokens.shape[0], time_cols, freq_bins)
    return block[:, delta_cols:, :].reshape(previous_tokens.shape[0], -1)


def _validate_window_prefixes(
    tokens_list: List[np.ndarray],
    start_frames: List[int],
    level_time_frames: int,
    level_grid: List[int],
    level_name: str,
) -> None:
    """
    Make sure windowed sampling copied the previous overlap exactly.

    The model should receive the previous overlapping code tail as immutable
    start_tokens. If this check fails, the generated windows are not aligned
    the same way they were sampled.
    """
    if len(tokens_list) <= 1:
        return
    if len(tokens_list) != len(start_frames):
        raise ValueError(
            f'{level_name} prefix validation got {len(tokens_list)} token blocks '
            f'but {len(start_frames)} start frames.'
        )

    checked_tokens = 0
    for idx in range(1, len(tokens_list)):
        expected = _extract_prefix_from_previous_window(
            previous_tokens=tokens_list[idx - 1],
            previous_start_frame=start_frames[idx - 1],
            current_start_frame=start_frames[idx],
            level_time_frames=level_time_frames,
            level_grid=level_grid,
        )
        if expected is None or expected.shape[1] == 0:
            continue

        actual = tokens_list[idx][:, :expected.shape[1]]
        if actual.shape != expected.shape:
            raise RuntimeError(
                f'{level_name} overlap prefix shape mismatch at window {idx}: '
                f'expected {expected.shape}, got {actual.shape}.'
            )

        mismatches = int(np.count_nonzero(actual != expected))
        if mismatches:
            raise RuntimeError(
                f'{level_name} overlap prefix mismatch at window {idx}: '
                f'{mismatches}/{expected.size} copied context tokens differ.'
            )
        checked_tokens += int(expected.size)

    print(f'{level_name.capitalize()} overlap prefix check: OK ({checked_tokens} copied tokens)')


def _compute_windowed_step(
    level_time_frames: int,
    level_grid: List[int],
    overlap_fraction: float,
) -> Tuple[int, float, int, int]:
    """Convert an overlap fraction into a token-column-aligned raw-frame hop."""
    if not 0.0 <= overlap_fraction < 1.0:
        raise ValueError(f'overlap_fraction must be in [0, 1), got {overlap_fraction}')

    time_cols, _, frames_per_token_col = _level_grid_info(level_time_frames, level_grid)
    overlap_cols = int(round(time_cols * overlap_fraction))
    overlap_cols = min(max(overlap_cols, 0), time_cols - 1)
    hop_cols = time_cols - overlap_cols
    step_frames = hop_cols * frames_per_token_col
    effective_overlap = overlap_cols / time_cols
    return step_frames, effective_overlap, overlap_cols, hop_cols


def _assemble_token_timeline(
    tokens_list: List[np.ndarray],
    start_frames: List[int],
    level_time_frames: int,
    level_grid: List[int],
    total_frames: int,
) -> np.ndarray:
    """Place generated token windows on a raw-frame-aligned token timeline."""
    if not tokens_list:
        return np.array([])
    if len(tokens_list) != len(start_frames):
        raise ValueError(f'tokens_list length ({len(tokens_list)}) != start_frames length ({len(start_frames)})')
    if not (isinstance(level_grid, (list, tuple)) and len(level_grid) == 2):
        raise ValueError(f'Expected level_grid=[time_cols, freq_bins], got {level_grid}')

    time_cols, freq_bins = int(level_grid[0]), int(level_grid[1])
    if level_time_frames % time_cols != 0:
        raise ValueError(
            f'level_time_frames={level_time_frames} is not divisible by time_cols={time_cols}'
        )
    frames_per_token_col = level_time_frames // time_cols
    total_time_cols = max(time_cols, math.ceil(total_frames / frames_per_token_col))

    batch_size = tokens_list[0].shape[0]
    timeline = np.zeros((batch_size, total_time_cols, freq_bins), dtype=tokens_list[0].dtype)

    for tokens, start_frame in zip(tokens_list, start_frames):
        block = tokens.reshape(batch_size, time_cols, freq_bins)
        start_col = int(round(int(start_frame) / frames_per_token_col))
        end_col = start_col + time_cols
        if start_col >= total_time_cols:
            continue
        block_end = min(time_cols, total_time_cols - start_col)
        timeline[:, start_col:start_col + block_end, :] = block[:, :block_end, :]

    return timeline.reshape(batch_size, total_time_cols * freq_bins)

def get_token_slice_for_frame(
    full_tokens: np.ndarray,
    start_frame: int,
    base_start_frame: int,
    level_time_frames: int,
    level_grid: List[int],
    slice_len: int,
) -> np.ndarray:
    """
    Extract a conditioning token slice aligned to a raw spectrogram start frame.

    Training slices are aligned by raw frame starts. This mirrors that logic by
    converting the requested frame offset into token time columns for the
    already-generated upper-level token stream.
    """
    if not (isinstance(level_grid, (list, tuple)) and len(level_grid) == 2):
        raise ValueError(f'Expected level_grid=[time_cols, freq_bins], got {level_grid}')

    time_cols, freq_bins = int(level_grid[0]), int(level_grid[1])
    if time_cols <= 0 or freq_bins <= 0:
        raise ValueError(f'Invalid level grid: {level_grid}')
    if level_time_frames % time_cols != 0:
        raise ValueError(
            f'level_time_frames={level_time_frames} is not divisible by time_cols={time_cols}'
        )

    frames_per_token_col = level_time_frames // time_cols
    relative_frame = int(start_frame) - int(base_start_frame)
    if relative_frame < 0:
        raise ValueError(
            f'Requested slice start_frame={start_frame} is before generated base_start_frame={base_start_frame}'
        )

    token_col_start = int(round(relative_frame / frames_per_token_col))
    start = token_col_start * freq_bins
    end = start + slice_len
    
    if start >= full_tokens.shape[1]:
        print(
            f'Warning: Requested slice start_frame={start_frame} is out of bounds '
            f'(token_start={start}, total_len={full_tokens.shape[1]}). Returning zeros.'
        )
        return np.zeros((full_tokens.shape[0], slice_len), dtype=full_tokens.dtype)
        
    actual_slice = full_tokens[:, start:end]
    
    if actual_slice.shape[1] < slice_len:
        print(
            f'Info: Slice at start_frame={start_frame} is shorter than expected '
            f'(got {actual_slice.shape[1]}, expected {slice_len}). Padding with zeros.'
        )
        pad_width = slice_len - actual_slice.shape[1]
        actual_slice = np.pad(actual_slice, ((0, 0), (0, pad_width)), mode='constant')
        
    return actual_slice
    
def _resolve_quantized_path(transformer_config: dict) -> Optional[str]:
    dataset_cfg = transformer_config.get('dataset', {}) if isinstance(transformer_config, dict) else {}
    quantized_path = dataset_cfg.get('quantized_data_path')
    if not quantized_path:
        return None
    return os.path.expanduser(str(quantized_path))


def _load_windowed_quantization_config(transformer_config: dict) -> Tuple[Dict, Optional[str]]:
    quantized_path = _resolve_quantized_path(transformer_config)
    if not quantized_path:
        return {}, None

    config_path = os.path.join(quantized_path, 'windowed_quantization_config.json')
    if not os.path.isfile(config_path):
        print(f'Warning: windowed_quantization_config.json not found at {config_path}; using transformer config fallbacks.')
        return {}, quantized_path

    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f), quantized_path


def assemble_spectrogram_chunks(
    spec_chunks: List[np.ndarray],
    start_frames: List[int],
    total_frames: int,
) -> np.ndarray:
    """Place decoded windows into the requested spectrogram timeline using linear crossfades."""
    if not spec_chunks:
        raise ValueError('No spectrogram chunks to assemble.')
    if len(spec_chunks) == 1:
        return spec_chunks[0][:, :, :total_frames, :].copy()
    if len(spec_chunks) != len(start_frames):
        raise ValueError(f'spec_chunks length ({len(spec_chunks)}) != start_frames length ({len(start_frames)})')

    first = spec_chunks[0]
    batch_size, freq_bins, _, channels = first.shape
    assembled = np.zeros((batch_size, freq_bins, total_frames, channels), dtype=np.float32)
    current_end = 0

    for chunk, start_frame in zip(spec_chunks, start_frames):
        start = int(start_frame)
        if start >= total_frames:
            continue
        usable = min(chunk.shape[2], total_frames - start)
        if usable <= 0:
            continue

        chunk = chunk[:, :, :usable, :].astype(np.float32, copy=False)
        end = start + usable

        if start >= current_end:
            assembled[:, :, start:end, :] = chunk
            current_end = max(current_end, end)
            continue

        overlap_end = min(current_end, end)
        overlap_len = max(0, overlap_end - start)
        if overlap_len > 0:
            fade_in = np.linspace(0.0, 1.0, num=overlap_len, dtype=np.float32).reshape(1, 1, overlap_len, 1)
            fade_out = 1.0 - fade_in
            assembled[:, :, start:overlap_end, :] = (
                assembled[:, :, start:overlap_end, :] * fade_out
                + chunk[:, :, :overlap_len, :] * fade_in
            )

        if end > overlap_end:
            chunk_start = overlap_len
            assembled[:, :, overlap_end:end, :] = chunk[:, :, chunk_start:chunk_start + (end - overlap_end), :]
        current_end = max(current_end, end)

    return assembled


def _decode_full_level_spectrogram(
    level: str,
    vqvae_ref: str,
    full_tokens: np.ndarray,
    level_grid: Optional[list],
    total_frames: int,
    device: torch.device,
    weights_file: str,
    save_dir: str,
) -> np.ndarray:
    if not (isinstance(level_grid, (list, tuple)) and len(level_grid) == 2):
        raise ValueError(f'{level} grid is required to decode full generated indices.')

    freq_bins = int(level_grid[1])
    if full_tokens.shape[1] % freq_bins != 0:
        raise ValueError(
            f'{level} full token length {full_tokens.shape[1]} is not divisible by freq_bins={freq_bins}.'
        )

    dynamic_grid = [full_tokens.shape[1] // freq_bins, freq_bins]
    print(f'Decoding full {level} token timeline with dynamic grid {dynamic_grid}...')
    vqvae = load_jukebox_model(vqvae_ref, level, device, weights_file)
    tokens_tensor = torch.from_numpy(full_tokens).to(device)
    decoded_specs = decode_jukebox_indices(vqvae, tokens_tensor, dynamic_grid, device)
    decoded_specs = decoded_specs[:, :, :total_frames, :]
    spectrogram_dir = save_level_spectrograms(
        decoded_specs,
        save_dir,
        level,
        root_subdir='spectrograms',
        npy_filename=f'{level}_full_decoded_specs.npy',
        filename_prefix=f'{level}_full_spectrogram',
        title_template=f'{level.capitalize()} full generated spectrogram {{index}}',
        cmap='magma',
        figsize=(12, 4),
    )
    print(f'Saved full {level} spectrograms to {spectrogram_dir}')

    del vqvae, tokens_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return decoded_specs


def _infer_slice_len(
    model_cfg: dict,
    target_seq_len: int,
    inferred_len_key: str,
    inferred_stride_key: str,
) -> int:
    inferred_len = int(model_cfg.get(inferred_len_key, 0))
    if inferred_len > 0:
        return inferred_len

    inferred_stride = int(model_cfg.get(inferred_stride_key, 0))
    if inferred_stride > 0:
        if target_seq_len % inferred_stride != 0:
            raise ValueError(
                f"Cannot infer conditioning slice length: target_seq_len={target_seq_len} "
                f"is not divisible by stride={inferred_stride} ({inferred_stride_key})."
            )
        return target_seq_len // inferred_stride

    raise ValueError(
        f"Missing both {inferred_len_key} and {inferred_stride_key} in saved model config."
    )

def _decode_bottom_blocks(
    vqvae,
    bottom_tokens_list: List[np.ndarray],
    bottom_grid: Optional[list],
    device: torch.device,
) -> List[np.ndarray]:
    reconstructed_spectrograms = []
    for bottom_tokens in bottom_tokens_list:
        bottom_tokens_tensor = torch.from_numpy(bottom_tokens).to(device)
        with torch.no_grad():
            decoded_specs = decode_jukebox_indices(vqvae, bottom_tokens_tensor, bottom_grid, device)
        reconstructed_spectrograms.append(decoded_specs)
    return reconstructed_spectrograms


def _save_audio_from_spectrogram(
    spectrograms: np.ndarray,
    min_max_values,
    save_dir: str,
    filename: str,
    hop_length: int,
    sample_rate: int,
    frame_size: int,
    spectrogram_type: str,
    n_mels: int,
    audio_method: str,
) -> str:
    """!
    @brief Convert decoded normalized spectrograms to audio and save the first sample.
    @param spectrograms Decoded spectrogram batch shaped `(B, F, T, 1)`.
    @param min_max_values Dataset min/max metadata used to denormalize the spectrogram.
    @param save_dir Generation output directory.
    @param filename Output filename inside the `audio` subdirectory.
    @param hop_length STFT/mel hop length used during preprocessing.
    @param sample_rate Audio sample rate.
    @param frame_size FFT frame size.
    @param spectrogram_type Spectrogram type: `linear` or `mel`.
    @param n_mels Number of mel bins when using mel spectrograms.
    @param audio_method Inversion method accepted by `SoundGenerator`.
    @return Full path of the written waveform file.
    """
    min_max_list = prepare_min_max_values(min_max_values, spectrograms.shape[0])
    sound_generator = SoundGenerator(
        None,
        hop_length=hop_length,
        sample_rate=sample_rate,
        n_fft=frame_size,
        spectrogram_type=spectrogram_type,
        n_mels=n_mels,
    )
    audio_signals = sound_generator.convert_spectrograms_to_audio(
        spectrograms,
        min_max_list,
        method=audio_method,
    )

    audio_dir = os.path.join(save_dir, 'audio')
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, filename)
    sf.write(audio_path, audio_signals[0], sample_rate)
    return audio_path


def main():
    parser = argparse.ArgumentParser(description='Generate music from hierarchical transformer priors.')
    parser.add_argument('--top_config', type=str, default=None, help='Path to top prior config.yaml or run directory')
    parser.add_argument('--middle_config', type=str, default=None, help='Path to middle prior config.yaml or run directory')
    parser.add_argument('--bottom_config', type=str, default=None, help='Path to bottom prior config.yaml or run directory')
    parser.add_argument('--top_run_root', type=str, default=DEFAULT_TOP_RUN_ROOT, help='Default top run root used when --top_config is not provided')
    parser.add_argument('--middle_run_root', type=str, default=DEFAULT_MIDDLE_RUN_ROOT, help='Default middle run root used when --middle_config is not provided')
    parser.add_argument('--bottom_run_root', type=str, default=DEFAULT_BOTTOM_RUN_ROOT, help='Default bottom run root used when --bottom_config is not provided')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature for all priors')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling (None disables top-k)')
    parser.add_argument('--weights_file', type=str, default='best_model.pth', help='Checkpoint filename for transformer priors (default: best_model.pth)')
    parser.add_argument(
        '--sampling_mode',
        type=str,
        default='windowed',
        choices=['windowed', 'independent'],
        help='Sampling strategy. windowed reuses overlapping previous codes as context (default).',
    )
    parser.add_argument(
        '--overlap_fraction',
        type=float,
        default=0.75,
        help='Overlap fraction for windowed sampling (default: 0.75).',
    )
    parser.add_argument(
        '--duration_seconds',
        type=float,
        default=30.0,
        help='Target generated song duration in seconds. This is also the timing conditioning total_duration_s.',
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible generation (set to negative to disable)')
    parser.add_argument('--audio_method', type=str, default='griffinlim', choices=['griffinlim', 'istft'], help='Spectrogram inversion method')
    parser.add_argument('--save_root', type=str, default='samples/generate_music_maestro', help='Root directory for generated outputs')
    args = parser.parse_args()

    if args.temperature <= 0:
        raise ValueError(f'--temperature must be > 0, got {args.temperature}')
    if args.top_k is not None and args.top_k < 0:
        raise ValueError(f'--top_k must be >= 0, got {args.top_k}')
    args.top_k = args.top_k if (args.top_k is not None and args.top_k > 0) else None
    if args.duration_seconds <= 0:
        raise ValueError(f'--duration_seconds must be > 0, got {args.duration_seconds}')
    if not 0.0 <= args.overlap_fraction < 1.0:
        raise ValueError(f'--overlap_fraction must be in [0, 1), got {args.overlap_fraction}')

    if args.seed is not None and args.seed < 0:
        args.seed = None
    set_global_seed(args.seed, deterministic=True)
    if args.seed is not None:
        print(f'Using deterministic seed: {args.seed}')

    try:
        save_dir = generate_hierarchical_music(args)
        print(f'Done! Saved generated samples to {save_dir}')
    except Exception as e:
        _debug(f'Generation failed in main with error: {type(e).__name__}: {e}')
        raise
def linear_crossfading(spec_chunk_1, spec_chunk_2, overlap_len):
    """
    Linearly crossfade between two spectrogram chunks over the specified overlap length
    Formula: S_blended = (1-alfa) * S_1 + alfa * S_2
    
    @spec_chunk_1: Spectrogram chunk from the first block (shape: [channels, time_frames, freq_bins])
    @spec_chunk_2: Spectrogram chunk from the second block (shape: [channels, time_frames, freq_bins])
    @overlap_len: Number of time frames to use for the crossfade (must be > 0)
    @returns: Blended spectrogram chunk with the same shape as the input chunks
    """
    assert overlap_len > 0, "Overlap length must be greater than 0 for crossfading"

    # Make copies to prevent in-place mutation bugs
    c1 = spec_chunk_1.copy()
    c2 = spec_chunk_2.copy()
    
    fade_in = np.linspace(0, 1, num=overlap_len).reshape(1, 1, overlap_len, 1)
    fade_out = 1 - fade_in

    # Apply crossfade to the overlapping regions in the Time dimension (index 2)
    blended_overlap = (c1[:, :, -overlap_len:] * fade_out) + (c2[:, :, :overlap_len] * fade_in)

    # Concatenate along the Time dimension (axis=2)
    combined = np.concatenate([
        c1[:, :, :-overlap_len], 
        blended_overlap, 
        c2[:, :, overlap_len:]
    ], axis=2)
    return combined


def generate_hierarchical_music(args) -> str:
    _debug('Resolving config paths...')
    top_transformer_prior_config_path = _resolve_prior_config_path(args.top_config, args.top_run_root, 'top')
    middle_transformer_prior_config_path = _resolve_prior_config_path(args.middle_config, args.middle_run_root, 'middle')
    bottom_transformer_prior_config_path = _resolve_prior_config_path(args.bottom_config, args.bottom_run_root, 'bottom')

    print(f'Top config: {top_transformer_prior_config_path}')
    print(f'Middle config: {middle_transformer_prior_config_path}')
    print(f'Bottom config: {bottom_transformer_prior_config_path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _debug(f'Using device: {device}')

    # Load the three trained priors.
    _debug('Loading transformer priors...')
    top_prior, top_config, _ = load_transformer_prior('top', top_transformer_prior_config_path, device, weights_file=args.weights_file)
    middle_prior, middle_config, _ = load_transformer_prior('middle', middle_transformer_prior_config_path, device, weights_file=args.weights_file)
    bottom_prior, bottom_config, _ = load_transformer_prior('bottom', bottom_transformer_prior_config_path, device, weights_file=args.weights_file)
    print(
        'Timing conditioning: '
        f"top={'enabled' if top_prior.use_timing_conditioning else 'disabled'}, "
        f"middle={'enabled' if middle_prior.use_timing_conditioning else 'disabled'}, "
        f"bottom={'enabled' if bottom_prior.use_timing_conditioning else 'disabled'} "
        '(learned absolute/relative/duration embeddings)'
    )

    top_seq_len = int(top_config['model']['inferred_seq_lens']['top'])
    middle_seq_len = int(middle_config['model']['inferred_seq_lens']['middle'])
    bottom_seq_len = int(bottom_config['model']['inferred_seq_lens']['bottom'])
    middle_model_cfg = middle_config.get('model', {}) if isinstance(middle_config, dict) else {}
    bottom_model_cfg = bottom_config.get('model', {}) if isinstance(bottom_config, dict) else {}
    top_grid = top_config['model'].get('inferred_grids', {}).get('top')
    middle_grid = middle_config['model'].get('inferred_grids', {}).get('middle')
    bottom_grid = bottom_config['model'].get('inferred_grids', {}).get('bottom')
    bottom_prior_cfg = bottom_config.get('priors', {}).get('bottom_prior', {}) if isinstance(bottom_config, dict) else {}
    bottom_second_cond_len = int(bottom_model_cfg.get('inferred_second_cond_seq_len', 0))
    bottom_condition_on_top = bool(bottom_prior_cfg.get('condition_on_top', False)) or bottom_second_cond_len > 0

    dataset_cfg = bottom_config.get('dataset', {}) if isinstance(bottom_config, dict) else {}
    quantization_cfg, quantized_path = _load_windowed_quantization_config(bottom_config)
    level_target_time_frames = dataset_cfg.get('level_target_time_frames') or {}
    top_tf = int(quantization_cfg.get('top_time_frames', level_target_time_frames.get('top', dataset_cfg.get('target_time_frames', 2048))))
    mid_tf = int(quantization_cfg.get('middle_time_frames', level_target_time_frames.get('middle', dataset_cfg.get('target_time_frames', 2048))))
    bot_tf = int(quantization_cfg.get('bottom_time_frames', level_target_time_frames.get('bottom', dataset_cfg.get('target_time_frames', 2048))))
    training_top_step = int(quantization_cfg.get('top_step_frames', top_tf))
    training_mid_step = int(quantization_cfg.get('middle_step_frames', mid_tf))
    training_bot_step = int(quantization_cfg.get('bottom_step_frames', bot_tf))
    for name, value in (
        ('top_time_frames', top_tf),
        ('middle_time_frames', mid_tf),
        ('bottom_time_frames', bot_tf),
        ('top_step_frames', training_top_step),
        ('middle_step_frames', training_mid_step),
        ('bottom_step_frames', training_bot_step),
    ):
        if value <= 0:
            raise ValueError(f'{name} must be > 0, got {value}')

    effective_overlap = None
    overlap_cols = {}
    hop_cols = {}
    if args.sampling_mode == 'windowed':
        top_step, top_overlap, top_overlap_cols, top_hop_cols = _compute_windowed_step(top_tf, top_grid, args.overlap_fraction)
        mid_step, mid_overlap, mid_overlap_cols, mid_hop_cols = _compute_windowed_step(mid_tf, middle_grid, args.overlap_fraction)
        bot_step, bot_overlap, bot_overlap_cols, bot_hop_cols = _compute_windowed_step(bot_tf, bottom_grid, args.overlap_fraction)
        effective_overlap = {'top': top_overlap, 'middle': mid_overlap, 'bottom': bot_overlap}
        overlap_cols = {'top': top_overlap_cols, 'middle': mid_overlap_cols, 'bottom': bot_overlap_cols}
        hop_cols = {'top': top_hop_cols, 'middle': mid_hop_cols, 'bottom': bot_hop_cols}
    else:
        top_step, mid_step, bot_step = training_top_step, training_mid_step, training_bot_step

    _debug('Resolving min_max_values.pkl path...')
    min_max_values_path = resolve_min_max_values_path(bottom_config, debug_fn=_debug)

    _debug('Loading bottom VQ-VAE config...')
    vqvae_cfg = bottom_config.get('vqvae', {}) if isinstance(bottom_config, dict) else {}
    bottom_vqvae_ref = vqvae_cfg['bottom_model_dir']
    middle_vqvae_ref = vqvae_cfg.get('middle_model_dir')
    top_vqvae_ref = vqvae_cfg.get('top_model_dir')
    vqvae_weights_file = vqvae_cfg.get('weights_file', 'best_model.pth')
    bottom_vqvae_config_path = resolve_vqvae_config_path(bottom_vqvae_ref)
    bottom_vqvae_config = load_config(bottom_vqvae_config_path)
    dataset_cfg = bottom_vqvae_config.get('dataset', {}) if isinstance(bottom_vqvae_config, dict) else {}

    sample_rate = int(quantization_cfg.get('sample_rate', dataset_cfg.get('sample_rate', SAMPLE_RATE)))
    hop_length = int(quantization_cfg.get('hop_length', dataset_cfg.get('hop_length', HOP_LENGTH)))
    frame_size = int(dataset_cfg.get('frame_size', FRAME_SIZE))
    spectrograms_path = dataset_cfg.get('processed_path', '')
    spectrogram_type_cfg = dataset_cfg.get('spectrogram_type')
    spectrogram_type = str(spectrogram_type_cfg).strip().lower() if spectrogram_type_cfg else (
        'mel' if 'mel' in str(spectrograms_path).lower() else 'linear'
    )
    n_mels = int(dataset_cfg.get('n_mels', 256))

    # Step 0: Prepare data
    ## Timing uses the requested song duration; windowed mode advances by overlap-controlled hops.
    
    ## Conditioning slice sizes are inferred from the exact training config saved with each run.
    mid_slice_len = _infer_slice_len(
        middle_model_cfg,
        target_seq_len=middle_seq_len,
        inferred_len_key='inferred_cond_seq_len',
        inferred_stride_key='inferred_upsample_stride',
    )
    bot_mid_slice_len = _infer_slice_len(
        bottom_model_cfg,
        target_seq_len=bottom_seq_len,
        inferred_len_key='inferred_cond_seq_len',
        inferred_stride_key='inferred_upsample_stride',
    )
    bot_top_slice_len = 0
    if bottom_condition_on_top:
        bot_top_slice_len = _infer_slice_len(
            bottom_model_cfg,
            target_seq_len=bottom_seq_len,
            inferred_len_key='inferred_second_cond_seq_len',
            inferred_stride_key='inferred_second_upsample_stride',
        )
    
    base_start_frame = 0
    total_source_frames = max(1, int(math.ceil((args.duration_seconds * sample_rate) / hop_length)))

    if args.sampling_mode == 'windowed':
        top_start_frames = build_windowed_starts(total_source_frames, top_tf, top_step)
        mid_start_frames = build_windowed_starts(total_source_frames, mid_tf, mid_step)
        bot_start_frames = build_windowed_starts(total_source_frames, bot_tf, bot_step)
    else:
        top_start_frames = build_level_starts(total_source_frames, top_tf, top_step)
        mid_start_frames = build_level_starts(total_source_frames, mid_tf, mid_step)
        bot_start_frames = build_level_starts(total_source_frames, bot_tf, bot_step)
    top_chunks = len(top_start_frames)
    middle_chunks = len(mid_start_frames)
    bottom_chunks = len(bot_start_frames)

    total_source_duration_s = (total_source_frames * hop_length) / sample_rate
    top_conditioning_frames = max(
        total_source_frames,
        max(mid_start_frames) + mid_tf,
        max(bot_start_frames) + bot_tf,
    )
    middle_conditioning_frames = max(
        total_source_frames,
        max(bot_start_frames) + bot_tf,
    )

    print(f"--- Generation Plan ---")
    print(f"Sampling mode: {args.sampling_mode}")
    if args.sampling_mode == 'windowed':
        print(
            "Requested/effective overlap: "
            f"{args.overlap_fraction:.2f} / "
            f"top={effective_overlap['top']:.3f}, middle={effective_overlap['middle']:.3f}, bottom={effective_overlap['bottom']:.3f}"
        )
    print(f"Requested generated duration: {args.duration_seconds:.2f}s")
    print(f"Timing total duration: {total_source_duration_s:.2f}s ({total_source_frames} frames)")
    print(f"Sampling window steps: Top={top_step}, Mid={mid_step}, Bot={bot_step} frames")
    print(f"Training window steps: Top={training_top_step}, Mid={training_mid_step}, Bot={training_bot_step} frames")
    if (
        args.sampling_mode == 'windowed'
        and (top_step, mid_step, bot_step) != (training_top_step, training_mid_step, training_bot_step)
    ):
        print(
            'Warning: sampling window steps do not match the quantized training dataset steps. '
            'For best quality, regenerate/retrain with matching top/middle/bottom step frames.'
        )
    print(f"Conditioning coverage frames: Top={top_conditioning_frames}, Middle={middle_conditioning_frames}")
    print(f"Chunks needed: Top={top_chunks}, Mid={middle_chunks}, Bot={bottom_chunks}")

    top_timing = build_timing_schedule(top_start_frames, hop_length, sample_rate, total_source_frames)
    mid_timing = build_timing_schedule(mid_start_frames, hop_length, sample_rate, total_source_frames)
    bot_timing = build_timing_schedule(bot_start_frames, hop_length, sample_rate, total_source_frames)

    # Step 1: Top-Level Unrolling (The Composer)
    ## Generate global structure block-by-block, reusing previous overlap in windowed mode.
    start_time = time.time()
    top_tokens_list = _generate_level_tokens(
        prior=top_prior,
        seq_len=top_seq_len,
        num_chunks=top_chunks,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        upper_tokens_list=None,
        timing_list=top_timing,
        start_frames=top_start_frames,
        level_time_frames=top_tf,
        level_grid=top_grid,
        use_windowed_prefix=args.sampling_mode == 'windowed',
    )
    if args.sampling_mode == 'windowed':
        _validate_window_prefixes(top_tokens_list, top_start_frames, top_tf, top_grid, 'top')
    full_top_tokens = _assemble_token_timeline(
        tokens_list=top_tokens_list,
        start_frames=top_start_frames,
        level_time_frames=top_tf,
        level_grid=top_grid,
        total_frames=top_conditioning_frames,
    )
    print('Top-level generation complete. Generated tokens for each block have shape:', top_tokens_list[0].shape if top_tokens_list else None)

    # Step 2: Hierarchical Upsampling (The Performers)
    ## conditioning middle level on the chunk of from the top level codes corresponding to the same segment
    top_slices_for_middle = [
        get_token_slice_for_frame(
            full_tokens=full_top_tokens,
            start_frame=start_frame,
            base_start_frame=base_start_frame,
            level_time_frames=top_tf,
            level_grid=top_grid,
            slice_len=mid_slice_len,
        )
        for start_frame in mid_start_frames
    ]
    middle_tokens_list = _generate_level_tokens(
        prior=middle_prior,
        seq_len=middle_seq_len,
        num_chunks=middle_chunks,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        upper_tokens_list=top_slices_for_middle,
        timing_list=mid_timing,
        start_frames=mid_start_frames,
        level_time_frames=mid_tf,
        level_grid=middle_grid,
        use_windowed_prefix=args.sampling_mode == 'windowed',
    )
    if args.sampling_mode == 'windowed':
        _validate_window_prefixes(middle_tokens_list, mid_start_frames, mid_tf, middle_grid, 'middle')
    full_middle_tokens = _assemble_token_timeline(
        tokens_list=middle_tokens_list,
        start_frames=mid_start_frames,
        level_time_frames=mid_tf,
        level_grid=middle_grid,
        total_frames=middle_conditioning_frames,
    )
    print('Middle-level generation complete. Generated tokens for each block have shape:', middle_tokens_list[0].shape if middle_tokens_list else None)
    print('Assembled full middle tokens shape:', full_middle_tokens.shape)
    print('Middle-level generation took: {:.2f} seconds'.format(time.time() - start_time))

    ## conditioning bottom level on the chunk of from the top and middle levels codes corresponding to the same segment
    middle_slices_for_bottom = [
        get_token_slice_for_frame(
            full_tokens=full_middle_tokens,
            start_frame=start_frame,
            base_start_frame=base_start_frame,
            level_time_frames=mid_tf,
            level_grid=middle_grid,
            slice_len=bot_mid_slice_len,
        )
        for start_frame in bot_start_frames
    ]
    top_slices_for_bottom = None
    if bottom_condition_on_top:
        top_slices_for_bottom = [
            get_token_slice_for_frame(
                full_tokens=full_top_tokens,
                start_frame=start_frame,
                base_start_frame=base_start_frame,
                level_time_frames=top_tf,
                level_grid=top_grid,
                slice_len=bot_top_slice_len,
            )
            for start_frame in bot_start_frames
        ]
    bottom_tokens_list = _generate_level_tokens(
        prior=bottom_prior,
        seq_len=bottom_seq_len,
        num_chunks=bottom_chunks,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        upper_tokens_list=middle_slices_for_bottom,
        second_upper_tokens_list=top_slices_for_bottom,
        timing_list=bot_timing,
        start_frames=bot_start_frames,
        level_time_frames=bot_tf,
        level_grid=bottom_grid,
        use_windowed_prefix=args.sampling_mode == 'windowed',
    )
    if args.sampling_mode == 'windowed':
        _validate_window_prefixes(bottom_tokens_list, bot_start_frames, bot_tf, bottom_grid, 'bottom')
    full_bottom_tokens = _assemble_token_timeline(
        tokens_list=bottom_tokens_list,
        start_frames=bot_start_frames,
        level_time_frames=bot_tf,
        level_grid=bottom_grid,
        total_frames=total_source_frames,
    )
    print('Generation complete. Bottom tokens length:', len(bottom_tokens_list),
          'with each block having shape:', bottom_tokens_list[0].shape if bottom_tokens_list else None)
    print('Assembled full bottom tokens shape:', full_bottom_tokens.shape)
    print('Generation took: {:.2f} seconds'.format(time.time() - start_time))

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join(args.save_root, current_time)
    os.makedirs(save_dir, exist_ok=True)
    timing_metadata = {
        'timing_source': 'duration_conditioned_window_schedule',
        'sampling_mode': args.sampling_mode,
        'spectrogram_assembly': 'linear_crossfade',
        'requested_overlap_fraction': float(args.overlap_fraction),
        'effective_overlap_fraction': effective_overlap,
        'overlap_time_cols': overlap_cols,
        'hop_time_cols': hop_cols,
        'quantized_path': quantized_path,
        'base_start_frame': int(base_start_frame),
        'requested_duration_seconds': float(args.duration_seconds),
        'total_source_frames': int(total_source_frames),
        'conditioning_coverage_frames': {
            'top': int(top_conditioning_frames),
            'middle': int(middle_conditioning_frames),
            'bottom': int(total_source_frames),
        },
        'sample_rate': int(sample_rate),
        'hop_length': int(hop_length),
        'level_time_frames': {'top': int(top_tf), 'middle': int(mid_tf), 'bottom': int(bot_tf)},
        'sampling_step_frames': {'top': int(top_step), 'middle': int(mid_step), 'bottom': int(bot_step)},
        'training_step_frames': {'top': int(training_top_step), 'middle': int(training_mid_step), 'bottom': int(training_bot_step)},
        'chunk_start_frames': {
            'top': [int(x) for x in top_start_frames],
            'middle': [int(x) for x in mid_start_frames],
            'bottom': [int(x) for x in bot_start_frames],
        },
        'full_token_shapes': {
            'top': list(full_top_tokens.shape),
            'middle': list(full_middle_tokens.shape),
            'bottom': list(full_bottom_tokens.shape),
        },
    }
    with open(os.path.join(save_dir, 'generation_timing_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(timing_metadata, f, indent=2)

    indices_dir = os.path.join(save_dir, 'indices')
    os.makedirs(indices_dir, exist_ok=True)
    np.save(os.path.join(indices_dir, 'top_full_indices.npy'), full_top_tokens.astype(np.int64, copy=False))
    np.save(os.path.join(indices_dir, 'middle_full_indices.npy'), full_middle_tokens.astype(np.int64, copy=False))
    np.save(os.path.join(indices_dir, 'bottom_full_indices.npy'), full_bottom_tokens.astype(np.int64, copy=False))
    np.savez_compressed(
        os.path.join(indices_dir, 'top_window_indices.npz'),
        **{f'window_{idx:04d}': tokens.astype(np.int64, copy=False) for idx, tokens in enumerate(top_tokens_list)},
    )
    np.savez_compressed(
        os.path.join(indices_dir, 'middle_window_indices.npz'),
        **{f'window_{idx:04d}': tokens.astype(np.int64, copy=False) for idx, tokens in enumerate(middle_tokens_list)},
    )
    np.savez_compressed(
        os.path.join(indices_dir, 'bottom_window_indices.npz'),
        **{f'window_{idx:04d}': tokens.astype(np.int64, copy=False) for idx, tokens in enumerate(bottom_tokens_list)},
    )
    print(f'Saved full generated token timelines to {indices_dir}')

    del top_prior, middle_prior, bottom_prior
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if top_vqvae_ref:
        _decode_full_level_spectrogram(
            level='top',
            vqvae_ref=top_vqvae_ref,
            full_tokens=full_top_tokens,
            level_grid=top_grid,
            total_frames=total_source_frames,
            device=device,
            weights_file=vqvae_weights_file,
            save_dir=save_dir,
        )
    if middle_vqvae_ref:
        _decode_full_level_spectrogram(
            level='middle',
            vqvae_ref=middle_vqvae_ref,
            full_tokens=full_middle_tokens,
            level_grid=middle_grid,
            total_frames=total_source_frames,
            device=device,
            weights_file=vqvae_weights_file,
            save_dir=save_dir,
        )

    print('Decoding bottom tokens into spectrograms...')
    _debug('Loading bottom VQ-VAE decoder...')
    vqvae_bottom_decoder = load_jukebox_model(
        bottom_vqvae_ref,
        'bottom',
        device,
        vqvae_weights_file,
    )
    vqvae_bottom_decoder.eval()

    # Step 3: Spectrogram Assembly (Audio Engineer)
    ## Assemble generated windows by raw frame starts; crossfade where windows overlap.
    decode_start_time = time.time()
    reconstructed_spectrograms = _decode_bottom_blocks(
        vqvae=vqvae_bottom_decoder,
        bottom_tokens_list=bottom_tokens_list,
        bottom_grid=bottom_grid,
        device=device,
    )
    print('Decoded spectrograms for all blocks. Each block has shape:', reconstructed_spectrograms[0].shape if reconstructed_spectrograms else None)
    print('Decoding took: {:.2f} seconds'.format(time.time() - decode_start_time))

    final_spectrogram = assemble_spectrogram_chunks(
        spec_chunks=reconstructed_spectrograms,
        start_frames=bot_start_frames,
        total_frames=total_source_frames,
    )
    print('Spectrogram reconstruction complete. Final spectrogram shape:', final_spectrogram.shape)

    # Step 4: Spectrogram Inversion (The Mastering Engineer)
    ## convert final spectrograms back to audio and save
    _debug(f'Loading min/max values from: {min_max_values_path}')
    with open(min_max_values_path, 'rb') as f:
        min_max_values = pickle.load(f)

    save_decoded_spectrograms(final_spectrogram, save_dir)
    bottom_audio_path = _save_audio_from_spectrogram(
        spectrograms=final_spectrogram,
        min_max_values=min_max_values,
        save_dir=save_dir,
        filename='sample.wav',
        hop_length=hop_length,
        sample_rate=sample_rate,
        frame_size=frame_size,
        spectrogram_type=spectrogram_type,
        n_mels=n_mels,
        audio_method=args.audio_method,
    )
    print(f'Saved bottom audio to {bottom_audio_path}')

    if args.save_middle_audio:
        if middle_decoded_specs is None:
            print('Skipping middle audio because the middle VQ-VAE reference is unavailable.')
        else:
            middle_audio_path = _save_audio_from_spectrogram(
                spectrograms=middle_decoded_specs,
                min_max_values=min_max_values,
                save_dir=save_dir,
                filename='middle_sample.wav',
                hop_length=hop_length,
                sample_rate=sample_rate,
                frame_size=frame_size,
                spectrogram_type=spectrogram_type,
                n_mels=n_mels,
                audio_method=args.audio_method,
            )
            print(f'Saved middle audio to {middle_audio_path}')

    return save_dir


if __name__ == '__main__':
    main()
