import os
import pickle
import sys
import soundfile as sf
import argparse
import random
import math
import json
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import numpy as np
import time

import torch
import matplotlib.pyplot as plt

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_config
from train_scripts.jukebox_utils import load_jukebox_model
from test_scripts.test_transformer_prior import load_transformer_prior
from generation.soundgenerator import SoundGenerator
from generation.transformer_io_utils import (
    prepare_min_max_values,
    resolve_min_max_values_path,
    resolve_vqvae_config_path,
    save_decoded_spectrograms,
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
    timing_list: Optional[List[np.ndarray]] = None,
) -> List[np.ndarray]:
    token_blocks = []

    for chunk in range(num_chunks):
        upper_indices = None
        if upper_tokens_list is not None:
            upper_indices = torch.from_numpy(upper_tokens_list[chunk]).to(device)

        second_upper_indices = None
        if second_upper_tokens_list is not None:
            second_upper_indices = torch.from_numpy(second_upper_tokens_list[chunk]).to(device)

        generate_kwargs = {
            'batch_size': 1,
            'start_tokens': None,
            'upper_indices': upper_indices,
            'second_upper_indices': second_upper_indices,
            'seq_len': seq_len,
            'temperature': temperature,
            'top_k': top_k,
            'device': device,
        }
        if timing_list is not None:
            timing = torch.from_numpy(timing_list[chunk]).to(device)
            generate_kwargs['timing'] = timing

        with torch.no_grad():
            tokens = prior.generate(**generate_kwargs).cpu().numpy()

        token_blocks.append(tokens)

    return token_blocks

def _build_level_starts(total_frames: int, window_size: int, step: int) -> List[int]:
    """
    Match preprocess_quantization.py window starts.

    The last window is shifted left when needed so it ends exactly on the
    requested song duration instead of extending timing past the song end.
    """
    if window_size <= 0:
        raise ValueError(f'window_size must be > 0, got {window_size}')
    if step <= 0:
        raise ValueError(f'step must be > 0, got {step}')
    if total_frames <= window_size:
        return [0]

    starts = list(range(0, total_frames - window_size + 1, step))
    last_start = total_frames - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


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
    
def _build_timing_schedule(
    start_frames: List[int],
    hop_length: int,
    sample_rate: int,
    total_source_frames: int,
) -> List[np.ndarray]:
    """
    Build timing metadata exactly like windowed quantization preprocessing.

    Training stores [start_time_s, full_source_duration_s, fraction_elapsed],
    where full_source_duration_s comes from the original song length, not from
    the short generated clip length.
    """
    total_duration_s = (int(total_source_frames) * hop_length) / sample_rate
    timing_list = []
    for start_frame in start_frames:
        start_time_s = (int(start_frame) * hop_length) / sample_rate
        fraction = start_time_s / max(total_duration_s, 1e-6)
        timing_list.append(np.array([[start_time_s, total_duration_s, fraction]], dtype=np.float32))
    return timing_list


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
    """Place decoded bottom windows into the requested spectrogram timeline."""
    if not spec_chunks:
        raise ValueError('No spectrogram chunks to assemble.')
    if len(spec_chunks) == 1:
        return spec_chunks[0][:, :, :total_frames, :].copy()
    if len(spec_chunks) != len(start_frames):
        raise ValueError(f'spec_chunks length ({len(spec_chunks)}) != start_frames length ({len(start_frames)})')

    first = spec_chunks[0]
    batch_size, freq_bins, _, channels = first.shape
    assembled = np.zeros((batch_size, freq_bins, total_frames, channels), dtype=np.float32)
    counts = np.zeros((1, 1, total_frames, 1), dtype=np.float32)

    for chunk, start_frame in zip(spec_chunks, start_frames):
        start = int(start_frame)
        if start >= total_frames:
            continue
        usable = min(chunk.shape[2], total_frames - start)
        assembled[:, :, start:start + usable, :] += chunk[:, :, :usable, :].astype(np.float32, copy=False)
        counts[:, :, start:start + usable, :] += 1.0

    counts = np.maximum(counts, 1.0)
    return assembled / counts


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
            decoded_specs = _decode_bottom_indices(vqvae, bottom_tokens_tensor, bottom_grid, device)
        reconstructed_spectrograms.append(decoded_specs)
    return reconstructed_spectrograms

def _decode_bottom_indices(
    vqvae,
    indices: torch.Tensor,
    grid: Optional[list],
    device: torch.device,
) -> np.ndarray:
    if indices.ndim != 2:
        raise ValueError(f'Expected indices shape (B, T), got {tuple(indices.shape)}')
    if not (isinstance(grid, list) and len(grid) == 2):
        raise ValueError('Bottom grid is required to reshape indices into (H, W)')

    # grid[0] is now Time, grid[1] is now Frequency
    time_steps, freq_bins = int(grid[0]), int(grid[1])
    
    if time_steps * freq_bins != indices.shape[1]:
        raise ValueError(f'Grid {grid} does not match seq_len={indices.shape[1]}')

    # View as (Batch, Time, Freq), then transpose to (Batch, Freq, Time)
    idx_2d = indices.view(indices.shape[0], time_steps, freq_bins).transpose(1, 2).contiguous().long().to(device)
    
    vqvae.eval()
    with torch.no_grad():
        emb = vqvae.vq.embedding[idx_2d]  # (B, Freq, Time, D)
        z_q = emb.permute(0, 3, 1, 2).contiguous()  # (B, D, Freq, Time)
        x_hat = vqvae.decoder(z_q)
        if vqvae.activation_layer is not None:
            x_hat = vqvae.activation_layer(x_hat)

    return x_hat.detach().cpu().permute(0, 2, 3, 1).numpy()


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

    if args.seed is not None and args.seed < 0:
        args.seed = None
    _set_seed(args.seed)

    try:
        save_dir = generate_hierarchical_music(args)
        print(f'Done! Saved generated samples to {save_dir}')
    except Exception as e:
        _debug(f'Generation failed in main with error: {type(e).__name__}: {e}')
        raise


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'Using deterministic seed: {seed}')

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
    top_step = int(quantization_cfg.get('top_step_frames', top_tf))
    mid_step = int(quantization_cfg.get('middle_step_frames', mid_tf))
    bot_step = int(quantization_cfg.get('bottom_step_frames', bot_tf))
    for name, value in (
        ('top_time_frames', top_tf),
        ('middle_time_frames', mid_tf),
        ('bottom_time_frames', bot_tf),
        ('top_step_frames', top_step),
        ('middle_step_frames', mid_step),
        ('bottom_step_frames', bot_step),
    ):
        if value <= 0:
            raise ValueError(f'{name} must be > 0, got {value}')

    _debug('Resolving min_max_values.pkl path...')
    min_max_values_path = resolve_min_max_values_path(bottom_config, debug_fn=_debug)

    _debug('Loading bottom VQ-VAE decoder...')
    bottom_vqvae_ref = bottom_config['vqvae']['bottom_model_dir']
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

    vqvae_bottom_decoder = load_jukebox_model(
        bottom_vqvae_ref,
        'bottom',
        device,
        bottom_config['vqvae']['weights_file'],
    )
    vqvae_bottom_decoder.eval()

    # Step 0: Prepare data
    ## Temporal math mirrors the windowed quantized dataset used for training.
    
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

    top_start_frames = _build_level_starts(total_source_frames, top_tf, top_step)
    mid_start_frames = _build_level_starts(total_source_frames, mid_tf, mid_step)
    bot_start_frames = _build_level_starts(total_source_frames, bot_tf, bot_step)
    top_chunks = len(top_start_frames)
    middle_chunks = len(mid_start_frames)
    bottom_chunks = len(bot_start_frames)

    total_source_duration_s = (total_source_frames * hop_length) / sample_rate

    print(f"--- Generation Plan ---")
    print(f"Requested generated duration: {args.duration_seconds:.2f}s")
    print(f"Timing total duration: {total_source_duration_s:.2f}s ({total_source_frames} frames)")
    print(f"Training window steps: Top={top_step}, Mid={mid_step}, Bot={bot_step} frames")
    print(f"Chunks needed: Top={top_chunks}, Mid={middle_chunks}, Bot={bottom_chunks}")

    top_timing = _build_timing_schedule(top_start_frames, hop_length, sample_rate, total_source_frames)
    mid_timing = _build_timing_schedule(mid_start_frames, hop_length, sample_rate, total_source_frames)
    bot_timing = _build_timing_schedule(bot_start_frames, hop_length, sample_rate, total_source_frames)

    # Step 1: Top-Level Unrolling (The Composer)
    ## Generate global structure block-by-block using the same window starts as training.
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
    )
    full_top_tokens = _assemble_token_timeline(
        tokens_list=top_tokens_list,
        start_frames=top_start_frames,
        level_time_frames=top_tf,
        level_grid=top_grid,
        total_frames=total_source_frames,
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
    )
    full_middle_tokens = _assemble_token_timeline(
        tokens_list=middle_tokens_list,
        start_frames=mid_start_frames,
        level_time_frames=mid_tf,
        level_grid=middle_grid,
        total_frames=total_source_frames,
    )
    print('Middle-level generation complete. Generated tokens for each block have shape:', middle_tokens_list[0].shape if middle_tokens_list else None)
    print('Stitched full middle tokens shape:', full_middle_tokens.shape)
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
    )
    print('Generation complete. Bottom tokens length:', len(bottom_tokens_list),
          'with each block having shape:', bottom_tokens_list[0].shape if bottom_tokens_list else None)
    print('Generation took: {:.2f} seconds'.format(time.time() - start_time))
    print('Decoding bottom tokens into spectrograms...')

    # Step 3: Spectrogram Assembly (Audio Engineer)
    ## Use the same start frames as training; average only if the final coverage window overlaps.
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
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join(args.save_root, current_time)
    os.makedirs(save_dir, exist_ok=True)
    timing_metadata = {
        'timing_source': 'windowed_training_schedule',
        'quantized_path': quantized_path,
        'base_start_frame': int(base_start_frame),
        'requested_duration_seconds': float(args.duration_seconds),
        'total_source_frames': int(total_source_frames),
        'sample_rate': int(sample_rate),
        'hop_length': int(hop_length),
        'level_time_frames': {'top': int(top_tf), 'middle': int(mid_tf), 'bottom': int(bot_tf)},
        'level_step_frames': {'top': int(top_step), 'middle': int(mid_step), 'bottom': int(bot_step)},
        'chunk_start_frames': {
            'top': [int(x) for x in top_start_frames],
            'middle': [int(x) for x in mid_start_frames],
            'bottom': [int(x) for x in bot_start_frames],
        },
    }
    with open(os.path.join(save_dir, 'generation_timing_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(timing_metadata, f, indent=2)

    _debug(f'Loading min/max values from: {min_max_values_path}')
    with open(min_max_values_path, 'rb') as f:
        min_max_values = pickle.load(f)

    save_decoded_spectrograms(final_spectrogram, save_dir)
    min_max_list = prepare_min_max_values(min_max_values, final_spectrogram.shape[0])


    sound_generator = SoundGenerator(
        vqvae_bottom_decoder,
        hop_length=hop_length,
        sample_rate=sample_rate,
        n_fft=frame_size,
        spectrogram_type=spectrogram_type,
        n_mels=n_mels,
    )
    audio_signals = sound_generator.convert_spectrograms_to_audio(
        final_spectrogram, min_max_list, method=args.audio_method
    )
    final_audio = audio_signals[0]
    
    # Save the final waveform
    audio_dir = os.path.join(save_dir, 'audio')
    os.makedirs(audio_dir, exist_ok=True)
    sf.write(os.path.join(audio_dir, 'sample.wav'), final_audio, sample_rate)

    return save_dir


if __name__ == '__main__':
    main()
