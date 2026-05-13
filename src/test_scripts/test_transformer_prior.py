from typing import Tuple, Optional, List, Dict

import os
import sys
import argparse
import re
import json
from datetime import datetime
import pickle

import numpy as np
import torch
import soundfile as sf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from test_scripts.hierarchical_pixelcnn_common import resolve_model_paths
from utils import load_config, set_global_seed
from generation.transformer_io_utils import (
    decode_jukebox_token_timeline,
    prepare_min_max_values,
    resolve_vqvae_config_path,
    save_indices_with_visualizations,
    save_level_spectrograms,
)
from modeling.torch.transformer_prior_conditioned import TransformerPriorConditioned
from generation.soundgenerator import SoundGenerator
from processing.preprocess_audio import SAMPLE_RATE, HOP_LENGTH, FRAME_SIZE, N_MELS
from train_scripts.jukebox_utils import load_jukebox_model
from windowed_data_utils import (
    assemble_token_timeline,
    build_timing_tensor,
    build_timing_schedule_with_offset,
    dynamic_grid_for_tokens,
    extract_prefix_from_previous_window,
    get_token_slice_for_frame,
    level_grid_info,
    overlapped_subwindow_starts,
    seconds_to_frames,
    validate_window_prefixes,
)

def _extract_num_embeddings(state_dict: dict) -> int:
    for key in ('to_logits.weight', 'to_logits.weight_orig'):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    for key in ('token_embedding.weight', 'token_embedding.weight_orig'):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    raise KeyError('Could not infer num_embeddings from checkpoint state_dict')

def _extract_cond_num_embeddings(state_dict: dict) -> Optional[int]:
    for key in ('conditioner.token_embedding.weight', 'conditioner.token_embedding.weight_orig'):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    return None

def _extract_second_cond_num_embeddings(state_dict: dict) -> Optional[int]:
    for key in ('second_conditioner.token_embedding.weight', 'second_conditioner.token_embedding.weight_orig'):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    return None

def _checkpoint_uses_2d_conditioner(state_dict: dict, prefix: str = 'conditioner') -> Optional[bool]:
    dims = set()
    for key in (
        f'{prefix}.layers.0.conv.weight',
        f'{prefix}.layers.0.conv.weight_orig',
        f'{prefix}.upsample.weight',
        f'{prefix}.upsample.weight_orig',
    ):
        weight = state_dict.get(key)
        if weight is not None:
            dims.add(int(weight.ndim))
    if not dims:
        return None
    if 3 in dims and 4 in dims:
        raise RuntimeError(
            f'{prefix} checkpoint mixes 1D and 2D conditioner weights. '
            'Please retrain with the fixed 2D conditioner.'
        )
    return bool(4 in dims)

def _stride_product(value) -> int:
    if value is None:
        return 0
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f'Expected stride tuple/list of length 2, got {value}')
        return int(value[0]) * int(value[1])
    return int(value)

def _parse_stride(value, use_2d: bool):
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f'Expected stride tuple/list of length 2, got {value}')
        stride = (int(value[0]), int(value[1]))
        if stride[0] <= 0 or stride[1] <= 0:
            raise ValueError(f'Invalid 2D stride: {value}')
        return stride if use_2d else stride[0] * stride[1]
    stride = int(value)
    if stride <= 0:
        return None
    if use_2d:
        return None
    return stride

def _grid_freq_bins(inferred_grids: dict, level: str) -> Optional[int]:
    grid = inferred_grids.get(level) if isinstance(inferred_grids, dict) else None
    if isinstance(grid, (tuple, list)) and len(grid) == 2:
        return int(grid[1])
    return None

def _infer_cond_freq_bins(
    model_cfg: dict,
    inferred_grids: dict,
    cond_level: str,
    freq_key: str,
) -> Optional[int]:
    value = model_cfg.get(freq_key, 0)
    if value:
        return int(value)
    return _grid_freq_bins(inferred_grids, cond_level)

def _infer_2d_stride(
    model_cfg: dict,
    inferred_grids: dict,
    target_level: str,
    cond_level: str,
    stride_key: str,
    cond_seq_len_key: str,
    cond_time_key: str,
    cond_freq_key: str,
):
    parsed = _parse_stride(model_cfg.get(stride_key), use_2d=True)
    if parsed is not None:
        return parsed

    target_grid = inferred_grids.get(target_level) if isinstance(inferred_grids, dict) else None
    if not (isinstance(target_grid, (tuple, list)) and len(target_grid) == 2):
        return None

    target_time, target_freq = int(target_grid[0]), int(target_grid[1])
    cond_freq = _infer_cond_freq_bins(model_cfg, inferred_grids, cond_level, cond_freq_key)
    if cond_freq is None or cond_freq <= 0:
        return None

    cond_time = int(model_cfg.get(cond_time_key, 0) or 0)
    if cond_time <= 0:
        cond_seq_len = int(model_cfg.get(cond_seq_len_key, 0) or 0)
        if cond_seq_len > 0 and cond_seq_len % cond_freq == 0:
            cond_time = cond_seq_len // cond_freq
        else:
            cond_grid = inferred_grids.get(cond_level) if isinstance(inferred_grids, dict) else None
            if isinstance(cond_grid, (tuple, list)) and len(cond_grid) == 2:
                cond_time = int(cond_grid[0])

    if cond_time <= 0:
        return None
    if target_time % cond_time != 0 or target_freq % cond_freq != 0:
        raise ValueError(
            f'Cannot infer 2D conditioner stride for {cond_level}->{target_level}: '
            f'target_grid=({target_time}, {target_freq}), cond_grid=({cond_time}, {cond_freq})'
        )
    return (target_time // cond_time, target_freq // cond_freq)

def _infer_1d_stride(model_cfg: dict, inferred_seq_lens: dict, target_level: str, cond_level: str, stride_key: str):
    parsed = _parse_stride(model_cfg.get(stride_key), use_2d=False)
    if parsed is not None:
        return parsed

    target_len = int(inferred_seq_lens.get(target_level, 0))
    cond_len = int(inferred_seq_lens.get(cond_level, 0))
    if cond_len > 0 and target_len % cond_len == 0:
        return target_len // cond_len
    return None

def _extract_conditioner_block_count(state_dict: dict, prefix: str = 'conditioner') -> Optional[int]:
    pattern = re.compile(rf'^{re.escape(prefix)}\.layers\.(\d+)\.conv\.weight$')
    max_idx = -1
    for key in state_dict.keys():
        match = pattern.match(key)
        if not match:
            continue
        idx = int(match.group(1))
        if idx > max_idx:
            max_idx = idx
    if max_idx < 0:
        return None
    return max_idx + 1

def _extract_conditioner_width(state_dict: dict, prefix: str = 'conditioner') -> Optional[int]:
    for key in (f'{prefix}.token_embedding.weight', f'{prefix}.token_embedding.weight_orig'):
        if key in state_dict:
            return int(state_dict[key].shape[1])
    return None

def _extract_conditioner_kernel_size(state_dict: dict, prefix: str = 'conditioner') -> Optional[int]:
    for key in (f'{prefix}.layers.0.conv.weight', f'{prefix}.layers.0.conv.weight_orig'):
        if key in state_dict:
            return int(state_dict[key].shape[-1])
    return None

def _load_config_and_checkpoint(model_dir_or_file: str, weights_file: str):
    config_path, model_path = resolve_model_paths(model_dir_or_file, weights_file)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found at {config_path}')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found at {model_path}')

    config = load_config(config_path)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
    print(f'Loaded checkpoint from {model_path}.')
    return config, state_dict, model_path

def load_transformer_prior(
    model_layer: str, model_dir_or_file: str,
    device: torch.device, weights_file: str = 'best_model.pth',
    disable_timing_conditioning: bool = False,
) -> Tuple[TransformerPriorConditioned, dict, str]:
    assert model_layer in ['top', 'middle', 'bottom'], f'model_layer must be one of "top", "middle", or "bottom", got {model_layer}'

    config, state_dict, model_path = _load_config_and_checkpoint(model_dir_or_file, weights_file)

    model_cfg = config.get('model', {})
    inferred_seq_lens = model_cfg.get('inferred_seq_lens', {})
    seq_len = int(inferred_seq_lens.get(model_layer, 0))
    if seq_len <= 0:
        raise ValueError(f'Missing model.inferred_seq_lens for level={model_layer}.')

    priors = config.get('priors', {})
    prior_cfg = priors.get(f'{model_layer}_prior') if isinstance(priors, dict) else None
    if prior_cfg is None:
        raise ValueError(f"Missing priors.{model_layer}_prior in config.")

    inferred_num_embeddings = _extract_num_embeddings(state_dict)
    num_embeddings_cfg = int(prior_cfg.get('num_embeddings', 0))
    num_embeddings = inferred_num_embeddings if num_embeddings_cfg <= 0 else num_embeddings_cfg
    if num_embeddings != inferred_num_embeddings:
        print(f'Warning: num_embeddings from config ({num_embeddings}) does not match checkpoint ({inferred_num_embeddings}). Using checkpoint value for loading model.')
        num_embeddings = inferred_num_embeddings

    use_bos_token = bool(prior_cfg.get('use_bos_token', False))
    token_embedding_weight = state_dict.get('token_embedding.weight')
    if token_embedding_weight is None:
        token_embedding_weight = state_dict.get('token_embedding.weight_orig')

    to_logits_weight = state_dict.get('to_logits.weight')
    if to_logits_weight is None:
        to_logits_weight = state_dict.get('to_logits.weight_orig')

    if token_embedding_weight is not None and to_logits_weight is not None:
        token_vocab = int(token_embedding_weight.shape[0])
        output_vocab = int(to_logits_weight.shape[0])
        if token_vocab == output_vocab + 1:
            use_bos_token = True
        elif token_vocab == output_vocab:
            use_bos_token = False

    max_time_steps_cfg = int(prior_cfg.get('max_time_steps', 0))
    max_time_steps = max_time_steps_cfg if max_time_steps_cfg > 0 else 100

    deprecated_timing_keys = [
        key for key in (
            'time_embedding.weight',
            'time_embedding.weight_orig',
            'timing_proj.weight',
            'timing_proj.weight_orig',
            'timing_proj.bias',
            'timing_proj.bias_orig',
        )
        if key in state_dict
    ]
    if deprecated_timing_keys and not disable_timing_conditioning:
        raise RuntimeError(
            f'Transformer prior checkpoint {model_path} uses deprecated timing parameters: '
            f'{deprecated_timing_keys}. This test/generation path now supports only '
            f'learned_absolute_relative timing. Retrain this prior with the new timing config, '
            f'or pass --disable_timing_conditioning to load it without timing.'
        )
    if deprecated_timing_keys and disable_timing_conditioning:
        print(
            f'Ignoring deprecated timing parameters for {model_layer} prior because '
            f'--disable_timing_conditioning is set: {deprecated_timing_keys}'
        )

    checkpoint_has_learned_timing = (
        'absolute_timing_embedding.weight' in state_dict
        or 'relative_timing_embedding.weight' in state_dict
        or 'duration_timing_embedding.weight' in state_dict
    )
    use_timing_conditioning = False if disable_timing_conditioning else bool(
        prior_cfg.get('use_timing_conditioning', checkpoint_has_learned_timing)
    )

    inferred_grids = model_cfg.get('inferred_grids', {})
    primary_uses_2d = _checkpoint_uses_2d_conditioner(state_dict, prefix='conditioner')
    second_uses_2d = _checkpoint_uses_2d_conditioner(state_dict, prefix='second_conditioner')
    if model_layer != 'top':
        use_2d_conditioner = (
            bool(primary_uses_2d)
            if primary_uses_2d is not None
            else bool(model_cfg.get('use_2d_conditioner', True))
        )
    else:
        use_2d_conditioner = bool(model_cfg.get('use_2d_conditioner', True))
    if second_uses_2d is not None and second_uses_2d != use_2d_conditioner:
        raise RuntimeError(
            f'Mixed conditioner layouts are not supported in {model_path}: '
            f'conditioner 2D={use_2d_conditioner}, second_conditioner 2D={second_uses_2d}'
        )

    cond_num_embeddings = None
    cond_block_len = None
    upsample_stride = None
    second_cond_num_embeddings = None
    second_cond_block_len = None
    second_upsample_stride = None
    condition_on_top = bool(prior_cfg.get('condition_on_top', False)) if model_layer == 'bottom' else False
    if model_layer != 'top':
        cond_num_embeddings = int(prior_cfg.get('cond_num_embeddings', 0))
        if cond_num_embeddings <= 0:
            cond_num_embeddings = _extract_cond_num_embeddings(state_dict)
        cond_level = 'top' if model_layer == 'middle' else 'middle'

        if use_2d_conditioner:
            cond_block_len = _infer_cond_freq_bins(
                model_cfg,
                inferred_grids,
                cond_level,
                'inferred_cond_freq_bins',
            )
            upsample_stride = _infer_2d_stride(
                model_cfg,
                inferred_grids,
                model_layer,
                cond_level,
                'inferred_upsample_stride',
                'inferred_cond_seq_len',
                'inferred_cond_time_cols',
                'inferred_cond_freq_bins',
            )
        else:
            upsample_stride = _infer_1d_stride(
                model_cfg,
                inferred_seq_lens,
                model_layer,
                cond_level,
                'inferred_upsample_stride',
            )

        if model_layer == 'bottom':
            second_cond_num_embeddings = int(prior_cfg.get('second_cond_num_embeddings', 0))
            if second_cond_num_embeddings <= 0:
                second_cond_num_embeddings = _extract_second_cond_num_embeddings(state_dict)

            if use_2d_conditioner:
                second_cond_block_len = _infer_cond_freq_bins(
                    model_cfg,
                    inferred_grids,
                    'top',
                    'inferred_second_cond_freq_bins',
                )
                second_upsample_stride = _infer_2d_stride(
                    model_cfg,
                    inferred_grids,
                    model_layer,
                    'top',
                    'inferred_second_upsample_stride',
                    'inferred_second_cond_seq_len',
                    'inferred_second_cond_time_cols',
                    'inferred_second_cond_freq_bins',
                )
            else:
                second_upsample_stride = _infer_1d_stride(
                    model_cfg,
                    inferred_seq_lens,
                    model_layer,
                    'top',
                    'inferred_second_upsample_stride',
                )
            if second_cond_num_embeddings is not None and second_upsample_stride is not None:
                condition_on_top = True

    conditioner_width_inferred = _extract_conditioner_width(state_dict, prefix='conditioner')
    conditioner_blocks_inferred = _extract_conditioner_block_count(state_dict, prefix='conditioner')
    conditioner_kernel_inferred = _extract_conditioner_kernel_size(state_dict, prefix='conditioner')

    conditioner_residual_block_width = int(
        prior_cfg.get(
            'conditioner_residual_block_width',
            conditioner_width_inferred if conditioner_width_inferred is not None else 1024,
        )
    )
    conditioner_residual_blocks = int(
        prior_cfg.get(
            'conditioner_residual_blocks',
            conditioner_blocks_inferred if conditioner_blocks_inferred is not None else 16,
        )
    )
    conditioner_kernel_size = int(
        prior_cfg.get(
            'conditioner_kernel_size',
            conditioner_kernel_inferred if conditioner_kernel_inferred is not None else 3,
        )
    )
    conditioner_conv_channels = int(
        prior_cfg.get('conditioner_conv_channels', conditioner_residual_block_width)
    )
    conditioner_dilation_growth_rate = int(prior_cfg.get('conditioner_dilation_growth_rate', 3))
    conditioner_dilation_cycle = int(prior_cfg.get('conditioner_dilation_cycle', 8))

    prior_transformer = TransformerPriorConditioned(
        num_embeddings=num_embeddings,
        model_dim=int(prior_cfg['model_dim']),
        num_heads=int(prior_cfg['num_heads']),
        num_layers=int(prior_cfg['num_layers']),
        dim_feedforward=int(prior_cfg['dim_feedforward']),
        max_seq_len=seq_len,
        block_len=int(prior_cfg.get('block_len', 16)),
        max_time_steps=max_time_steps,
        is_upsampler=model_layer != 'top',
        cond_num_embeddings=cond_num_embeddings if model_layer != 'top' else None,
        cond_block_len=cond_block_len if model_layer != 'top' else None,
        upsample_stride=upsample_stride if model_layer != 'top' else None,
        second_cond_num_embeddings=second_cond_num_embeddings if condition_on_top else None,
        second_cond_block_len=second_cond_block_len if condition_on_top else None,
        second_upsample_stride=second_upsample_stride if condition_on_top else None,
        conditioner_residual_block_width=conditioner_residual_block_width,
        conditioner_residual_blocks=conditioner_residual_blocks,
        conditioner_kernel_size=conditioner_kernel_size,
        conditioner_conv_channels=conditioner_conv_channels,
        conditioner_dilation_growth_rate=conditioner_dilation_growth_rate,
        conditioner_dilation_cycle=conditioner_dilation_cycle,
        use_bos_token=use_bos_token,
        use_timing_conditioning=use_timing_conditioning,
        timing_num_bins=int(prior_cfg.get('timing_num_bins', 1024)),
        duration_num_bins=int(prior_cfg.get('duration_num_bins', 256)),
        timing_window_seconds=prior_cfg.get('timing_window_seconds', model_cfg.get('timing_window_seconds')),
        timing_max_duration_seconds=float(prior_cfg.get('timing_max_duration_seconds', 3600.0)),
        timing_embedding_init_std=float(prior_cfg.get('timing_embedding_init_std', 0.02)),
        timing_embedding_scale=float(prior_cfg.get('timing_embedding_scale', 1.0)),
        use_2d_conditioner=use_2d_conditioner,
        attention_qkv_ratio=float(prior_cfg.get('attention_qkv_ratio', 1.0)),
        dropout=float(prior_cfg.get('dropout', 0.1)),
    ).to(device)

    load_state = dict(state_dict)
    if not use_timing_conditioning:
        for key in (
            'absolute_timing_embedding.weight',
            'relative_timing_embedding.weight',
            'duration_timing_embedding.weight',
            'time_embedding.weight',
            'time_embedding.weight_orig',
            'timing_proj.weight',
            'timing_proj.weight_orig',
            'timing_proj.bias',
            'timing_proj.bias_orig',
        ):
            load_state.pop(key, None)

    missing, unexpected = prior_transformer.load_state_dict(load_state, strict=False)
    allowed_missing = set()
    if not use_timing_conditioning:
        allowed_missing.update({
            'absolute_timing_embedding.weight',
            'relative_timing_embedding.weight',
            'duration_timing_embedding.weight',
        })
    missing = [key for key in missing if key not in allowed_missing]
    if missing or unexpected:
        raise RuntimeError(
            f'Error loading Transformer prior from {model_path}. '
            f'Missing keys: {missing[:10]} Unexpected keys: {unexpected[:10]}'
        )
    prior_transformer.eval()

    return prior_transformer, config, model_path

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

def _resolve_decode_context_cols(requested_context_cols: int, chunk_time_cols: int) -> int:
    if requested_context_cols < 0:
        return max(1, int(chunk_time_cols) // 2)
    return int(requested_context_cols)

def _infer_slice_len(
    model_cfg: dict,
    target_seq_len: int,
    inferred_len_key: str,
    inferred_stride_key: str,
) -> int:
    inferred_len = int(model_cfg.get(inferred_len_key, 0))
    if inferred_len > 0:
        return inferred_len

    inferred_stride = _stride_product(model_cfg.get(inferred_stride_key, 0))
    if inferred_stride > 0:
        if target_seq_len % inferred_stride != 0:
            raise ValueError(
                f'Cannot infer conditioning slice length: target_seq_len={target_seq_len} '
                f'is not divisible by stride={inferred_stride} ({inferred_stride_key}).'
            )
        return target_seq_len // inferred_stride

    raise ValueError(
        f'Missing both {inferred_len_key} and {inferred_stride_key} in saved model config.'
    )

def _top_window_from_quantized_payload(
    payload: dict,
    top_time_frames: int,
    top_grid: List[int],
    sample_rate: int,
    hop_length: int,
    requested_start_frame: int = 0,
) -> Tuple[np.ndarray, int, int, torch.Tensor]:
    if 'top' not in payload:
        raise KeyError('Quantized payload does not contain a "top" entry.')

    top_array = np.asarray(payload['top'])
    if top_array.ndim != 2:
        raise ValueError(f'Expected payload["top"] to have shape (time_cols, freq_bins), got {top_array.shape}')

    time_cols, freq_bins, frames_per_token_col = level_grid_info(top_time_frames, top_grid)
    payload_freq_bins = int(top_array.shape[1])
    if payload_freq_bins != freq_bins:
        raise ValueError(
            f'Real top quantized payload freq bins ({payload_freq_bins}) do not match the top prior grid ({freq_bins}).'
        )

    payload_format = payload.get('format')
    if payload_format == 'windowed_v1':
        base_start_frame = int(payload.get('start_frame', 0))
        total_source_frames = int(payload.get('total_frames', top_time_frames))
        if requested_start_frame not in (0, base_start_frame):
            print(
                f'Warning: ignoring --real_top_start_frame={requested_start_frame} for windowed payload; '
                f'using payload start_frame={base_start_frame}.'
            )
        top_window = top_array
        if int(top_window.shape[0]) != time_cols:
            if int(top_window.shape[0]) > time_cols:
                top_window = top_window[:time_cols, :]
            else:
                pad_rows = time_cols - int(top_window.shape[0])
                top_window = np.pad(top_window, ((0, pad_rows), (0, 0)), mode='constant')
        if 'timing' in payload:
            timing = torch.as_tensor(payload['timing'], dtype=torch.float32)
        else:
            timing = build_timing_tensor(base_start_frame, total_source_frames, sample_rate, hop_length)
        return top_window.astype(np.int64, copy=False), base_start_frame, total_source_frames, timing

    total_source_frames = int(payload.get('total_frames', top_time_frames))
    max_start_frame = max(0, total_source_frames - top_time_frames)
    base_start_frame = max(0, min(int(requested_start_frame), max_start_frame))
    if int(requested_start_frame) != base_start_frame:
        print(
            f'Adjusted --real_top_start_frame from {requested_start_frame} to {base_start_frame} '
            f'to fit within total_frames={total_source_frames}.'
        )

    token_col_start = int(round(base_start_frame / frames_per_token_col))
    top_window = top_array[token_col_start:token_col_start + time_cols, :]
    if int(top_window.shape[0]) < time_cols:
        pad_rows = time_cols - int(top_window.shape[0])
        top_window = np.pad(top_window, ((0, pad_rows), (0, 0)), mode='constant')

    timing = build_timing_tensor(base_start_frame, total_source_frames, sample_rate, hop_length)
    return top_window.astype(np.int64, copy=False), base_start_frame, total_source_frames, timing

def _load_real_top_conditioning(
    quantized_path: str,
    num_samples: int,
    top_time_frames: int,
    top_grid: List[int],
    sample_rate: int,
    hop_length: int,
    requested_start_frame: int = 0,
) -> Dict[str, object]:
    resolved_path = os.path.abspath(os.path.expanduser(quantized_path))
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f'--real_top_quantized not found: {resolved_path}')

    payload = torch.load(resolved_path, map_location='cpu', weights_only=False)
    top_window, base_start_frame, total_source_frames, timing = _top_window_from_quantized_payload(
        payload=payload,
        top_time_frames=top_time_frames,
        top_grid=top_grid,
        sample_rate=sample_rate,
        hop_length=hop_length,
        requested_start_frame=requested_start_frame,
    )

    flat_top = top_window.reshape(1, -1)
    repeated_top = np.repeat(flat_top, int(num_samples), axis=0).astype(np.int64, copy=False)
    repeated_timing = timing.unsqueeze(0).repeat(int(num_samples), 1)
    return {
        'tokens': repeated_top,
        'timing': repeated_timing,
        'base_start_frame': int(base_start_frame),
        'total_source_frames': int(total_source_frames),
        'source_path': resolved_path,
    }

def _generate_level_windows(
    prior,
    seq_len: int,
    num_samples: int,
    start_frames: List[int],
    device: torch.device,
    temperature: float,
    top_k: Optional[int],
    upper_tokens_list: Optional[List[np.ndarray]] = None,
    second_upper_tokens_list: Optional[List[np.ndarray]] = None,
    timing_list: Optional[List[torch.Tensor]] = None,
    level_name: str = 'level',
    progress_interval: int = 128,
    level_time_frames: Optional[int] = None,
    level_grid: Optional[List[int]] = None,
    use_overlap_prefixes: bool = False,
) -> List[np.ndarray]:
    token_blocks = []
    previous_tokens = None
    previous_start_frame = None

    for chunk, start_frame in enumerate(start_frames):
        upper_indices = None
        if upper_tokens_list is not None:
            upper_indices = torch.from_numpy(upper_tokens_list[chunk]).to(device)

        second_upper_indices = None
        if second_upper_tokens_list is not None:
            second_upper_indices = torch.from_numpy(second_upper_tokens_list[chunk]).to(device)

        start_tokens = None
        if use_overlap_prefixes and previous_tokens is not None:
            if level_time_frames is None or level_grid is None:
                raise ValueError('level_time_frames and level_grid are required when overlap prefixes are enabled.')
            prefix = extract_prefix_from_previous_window(
                previous_tokens=previous_tokens,
                previous_start_frame=previous_start_frame,
                current_start_frame=start_frame,
                level_time_frames=level_time_frames,
                level_grid=level_grid,
            )
            if prefix is not None and prefix.shape[1] > 0:
                start_tokens = torch.from_numpy(prefix).to(device)

        generate_kwargs = {
            'batch_size': num_samples,
            'start_tokens': start_tokens,
            'upper_indices': upper_indices,
            'second_upper_indices': second_upper_indices,
            'seq_len': seq_len,
            'temperature': temperature,
            'top_k': top_k,
            'device': device,
            'progress_label': f'{level_name} window {chunk + 1}/{len(start_frames)}',
            'progress_interval': progress_interval,
        }
        if timing_list is not None:
            generate_kwargs['timing'] = timing_list[chunk].to(device=device, dtype=torch.float32)

        with torch.no_grad():
            tokens = prior.generate(**generate_kwargs).cpu().numpy()

        token_blocks.append(tokens)
        previous_tokens = tokens
        previous_start_frame = start_frame

    return token_blocks

def _resolve_level_vqvae_path(level: str, bottom_vqvae_path: Optional[str], vqvae_cfg: dict) -> Optional[str]:
    if level == 'bottom':
        return bottom_vqvae_path or vqvae_cfg.get('bottom_model_dir')
    return vqvae_cfg.get(f'{level}_model_dir')

def _resolve_min_max_values_path(
    resolved_dataset_cfg: dict,
    vqvae_config_path: str,
    override_path: Optional[str],
) -> str:
    if override_path:
        candidate = os.path.abspath(os.path.expanduser(override_path))
        if os.path.isdir(candidate):
            candidate = os.path.join(candidate, 'min_max_values.pkl')
        if not os.path.exists(candidate):
            raise FileNotFoundError(f'--min_max_values_path not found: {candidate}')
        return candidate

    min_max_values_path = resolved_dataset_cfg.get('min_max_values_path')
    if not min_max_values_path:
        raise ValueError(f'Missing dataset.min_max_values_path in VQ-VAE config: {vqvae_config_path}')
    if not os.path.exists(min_max_values_path):
        raise FileNotFoundError(f'min_max_values.pkl not found at {min_max_values_path}')
    return min_max_values_path

def _decode_level_to_audio(
    level: str,
    tokens: torch.Tensor,
    grid: Optional[list],
    vqvae_path: str,
    weights_file: str,
    audio_method: str,
    save_dir: str,
    device: torch.device,
    chunk_time_cols: Optional[int] = None,
    decode_context_cols: int = 0,
    trim_frames: Optional[int] = None,
    min_max_values_path_override: Optional[str] = None,
) -> None:
    vqvae_config_path = resolve_vqvae_config_path(vqvae_path)
    vqvae_config = load_config(vqvae_config_path)
    resolved_dataset_cfg = vqvae_config.get('dataset', {}) if isinstance(vqvae_config, dict) else {}

    min_max_values_path = _resolve_min_max_values_path(
        resolved_dataset_cfg=resolved_dataset_cfg,
        vqvae_config_path=vqvae_config_path,
        override_path=min_max_values_path_override,
    )

    sample_rate = int(resolved_dataset_cfg.get('sample_rate', SAMPLE_RATE))
    hop_length = int(resolved_dataset_cfg.get('hop_length', HOP_LENGTH))
    frame_size = int(resolved_dataset_cfg.get('frame_size', FRAME_SIZE))
    spectrograms_path = resolved_dataset_cfg.get('processed_path', '')
    spectrogram_type_cfg = resolved_dataset_cfg.get('spectrogram_type')
    spectrogram_type = str(spectrogram_type_cfg).strip().lower() if spectrogram_type_cfg else (
        'mel' if 'mel' in str(spectrograms_path).lower() else 'linear'
    )
    n_mels = int(resolved_dataset_cfg.get('n_mels', N_MELS))

    print(f'Loading {level} VQ-VAE from {vqvae_path}')
    vqvae = load_jukebox_model(vqvae_path, level, device, weights_file)

    with open(min_max_values_path, 'rb') as f:
        min_max_values = pickle.load(f)

    print(f'Decoding {level} indices into spectrograms and audio...')
    decoded_specs = decode_jukebox_token_timeline(
        vqvae=vqvae,
        tokens=tokens,
        grid=grid,
        device=device,
        chunk_time_cols=chunk_time_cols,
        context_cols=decode_context_cols,
        trim_frames=trim_frames,
    )
    spectrogram_dir = save_level_spectrograms(
        decoded_specs,
        os.path.join(save_dir, level),
        level,
        root_subdir='spectrograms',
        npy_filename=f'{level}_decoded_specs.npy',
        filename_prefix=f'{level}_spectrogram',
        title_template=f'{level.capitalize()} decoded spectrogram {{index}}',
        cmap='magma',
        figsize=(10, 4),
    )
    min_max_list = prepare_min_max_values(min_max_values, decoded_specs.shape[0])

    sound_generator = SoundGenerator(
        vqvae,
        hop_length=hop_length,
        sample_rate=sample_rate,
        n_fft=frame_size,
        spectrogram_type=spectrogram_type,
        n_mels=n_mels,
    )
    audio_signals = sound_generator.convert_spectrograms_to_audio(
        decoded_specs, min_max_list, method=audio_method
    )

    audio_dir = os.path.join(save_dir, level, 'audio')
    os.makedirs(audio_dir, exist_ok=True)
    for i, signal in enumerate(audio_signals):
        sf.write(os.path.join(audio_dir, f'{level}_sample_{i}.wav'), signal, sample_rate)

    print(f'Saved {level} spectrograms to {spectrogram_dir}')
    print(f'Saved {level} audio to {audio_dir}')

    del vqvae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def test_transformer_prior(
    top_prior_path: str,
    middle_prior_path: str,
    bottom_prior_path: str,
    bottom_vqvae_path: Optional[str],
    audio_method: str,
    num_samples: int,
    temperature: float,
    top_k: int,
    weights_file: str,
    decode_level: str = 'all',
    timing: Optional[torch.Tensor] = None,
    full_length: bool = False,
    full_length_until: str = 'auto',
    generate_until: str = 'auto',
    progress_interval: int = 128,
    decode_context_cols: int = -1,
    min_max_values_path: Optional[str] = None,
    disable_timing_conditioning: bool = False,
    full_length_overlap_fraction: float = 0.5,
    timing_duration_seconds: float = 240.0,
    seed: Optional[int] = 42,
    real_top_quantized_path: Optional[str] = None,
    real_top_start_frame: int = 0,
):
    if seed is not None:
        set_global_seed(int(seed), deterministic=True)
        print(f'Using deterministic seed: {int(seed)}')
    if timing_duration_seconds <= 0:
        raise ValueError(f'--timing_duration_seconds must be > 0, got {timing_duration_seconds}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    top_k_value = top_k if (top_k is not None and top_k > 0) else None

    print(f'Loading top Transformer prior from {top_prior_path}')
    top_prior, top_config, _ = load_transformer_prior(
        'top',
        top_prior_path,
        device,
        weights_file,
        disable_timing_conditioning=disable_timing_conditioning,
    )
    top_seq_len = int(top_config['model']['inferred_seq_lens']['top'])
    top_grid = top_config['model'].get('inferred_grids', {}).get('top')

    print(f'Loading middle Transformer prior from {middle_prior_path}')
    middle_prior, middle_config, _ = load_transformer_prior(
        'middle',
        middle_prior_path,
        device,
        weights_file,
        disable_timing_conditioning=disable_timing_conditioning,
    )
    middle_seq_len = int(middle_config['model']['inferred_seq_lens']['middle'])
    middle_grid = middle_config['model'].get('inferred_grids', {}).get('middle')

    print(f'Loading bottom Transformer prior from {bottom_prior_path}')
    bottom_prior, bottom_config, _ = load_transformer_prior(
        'bottom',
        bottom_prior_path,
        device,
        weights_file,
        disable_timing_conditioning=disable_timing_conditioning,
    )
    bottom_seq_len = int(bottom_config['model']['inferred_seq_lens']['bottom'])
    bottom_grid = bottom_config['model'].get('inferred_grids', {}).get('bottom')
    bottom_prior_cfg = bottom_config.get('priors', {}).get('bottom_prior', {}) if isinstance(bottom_config, dict) else {}
    bottom_condition_on_top = bool(bottom_prior_cfg.get('condition_on_top', False)) or getattr(bottom_prior, 'second_conditioner', None) is not None

    vqvae_cfg = bottom_config.get('vqvae', {}) if isinstance(bottom_config, dict) else {}
    effective_bottom_vqvae = bottom_vqvae_path or vqvae_cfg.get('bottom_model_dir')
    effective_vqvae_weights_file = vqvae_cfg.get('weights_file', 'best_model.pth')

    decode_level = str(decode_level).strip().lower()
    if decode_level not in ('all', 'bottom', 'middle', 'top'):
        raise ValueError(f'--decode_level must be one of "all", "bottom", "middle", "top", got {decode_level}')
    if generate_until not in ('auto', 'top', 'middle', 'bottom'):
        raise ValueError(f'--generate_until must be one of "auto", "top", "middle", "bottom", got {generate_until}')
    if progress_interval < 0:
        raise ValueError(f'--progress_interval must be >= 0, got {progress_interval}')

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join('samples', 'transformer_prior_maestro', current_time)
    os.makedirs(save_dir, exist_ok=True)

    if full_length:
        if decode_context_cols < -1:
            raise ValueError(f'--decode_context_cols must be >= -1, got {decode_context_cols}')
        if full_length_overlap_fraction < 0.0 or full_length_overlap_fraction >= 1.0:
            raise ValueError(
                f'--full_length_overlap_fraction must be in [0, 1), got {full_length_overlap_fraction}'
            )

        depth_by_level = {'top': 0, 'middle': 1, 'bottom': 2, 'all': 2}
        if full_length_until == 'auto':
            target_depth = depth_by_level[decode_level]
        else:
            target_depth = depth_by_level[full_length_until]

        transformer_dataset_cfg = bottom_config.get('dataset', {}) if isinstance(bottom_config, dict) else {}
        dataset_cfg = dict(transformer_dataset_cfg)
        if effective_bottom_vqvae:
            vqvae_config_path = resolve_vqvae_config_path(effective_bottom_vqvae)
            vqvae_config = load_config(vqvae_config_path)
            if isinstance(vqvae_config, dict):
                dataset_cfg = {**vqvae_config.get('dataset', {}), **dataset_cfg}

        quantization_cfg, _ = _load_windowed_quantization_config(bottom_config)
        level_target_time_frames = transformer_dataset_cfg.get('level_target_time_frames') or {}
        top_tf = int(quantization_cfg.get('top_time_frames', level_target_time_frames.get('top', transformer_dataset_cfg.get('target_time_frames', 2048))))
        mid_tf = int(quantization_cfg.get('middle_time_frames', level_target_time_frames.get('middle', transformer_dataset_cfg.get('target_time_frames', 2048))))
        bot_tf = int(quantization_cfg.get('bottom_time_frames', level_target_time_frames.get('bottom', transformer_dataset_cfg.get('target_time_frames', 2048))))
        sample_rate = int(quantization_cfg.get('sample_rate', dataset_cfg.get('sample_rate', SAMPLE_RATE)))
        hop_length = int(quantization_cfg.get('hop_length', dataset_cfg.get('hop_length', HOP_LENGTH)))
        real_top_conditioning = None
        if real_top_quantized_path:
            real_top_conditioning = _load_real_top_conditioning(
                quantized_path=real_top_quantized_path,
                num_samples=num_samples,
                top_time_frames=top_tf,
                top_grid=top_grid,
                sample_rate=sample_rate,
                hop_length=hop_length,
                requested_start_frame=real_top_start_frame,
            )

        for name, value in (
            ('top_time_frames', top_tf),
            ('middle_time_frames', mid_tf),
            ('bottom_time_frames', bot_tf),
        ):
            if value <= 0:
                raise ValueError(f'{name} must be > 0, got {value}')

        _, _, top_frames_per_token_col = level_grid_info(top_tf, top_grid)
        timing_total_frames = (
            int(real_top_conditioning['total_source_frames'])
            if real_top_conditioning is not None
            else seconds_to_frames(timing_duration_seconds, sample_rate, hop_length)
        )
        generated_span_frames = top_tf
        base_start_frame = (
            int(real_top_conditioning['base_start_frame'])
            if real_top_conditioning is not None
            else 0
        )
        top_start_frames = [0]
        mid_start_frames = (
            overlapped_subwindow_starts(
                top_tf,
                mid_tf,
                full_length_overlap_fraction,
                'middle',
                align_frames=top_frames_per_token_col,
            )
            if target_depth >= 1 else []
        )
        bot_start_frames = (
            overlapped_subwindow_starts(
                top_tf,
                bot_tf,
                full_length_overlap_fraction,
                'bottom',
                align_frames=top_frames_per_token_col,
            )
            if target_depth >= 2 else []
        )
        top_conditioning_frames = top_tf
        middle_conditioning_frames = top_tf
        middle_chunks_per_top = len(mid_start_frames)
        bottom_chunks_per_middle = (mid_tf // bot_tf) if target_depth >= 2 and mid_tf % bot_tf == 0 else None

        timing_enabled = {
            'top': top_prior.use_timing_conditioning and not disable_timing_conditioning,
            'middle': middle_prior.use_timing_conditioning and not disable_timing_conditioning and target_depth >= 1,
            'bottom': bottom_prior.use_timing_conditioning and not disable_timing_conditioning and target_depth >= 2,
        }
        top_timing = (
            build_timing_schedule_with_offset(
                top_start_frames,
                hop_length,
                sample_rate,
                timing_total_frames,
                base_start_frame=base_start_frame,
            )
            if timing_enabled['top'] else None
        )
        mid_timing = (
            build_timing_schedule_with_offset(
                mid_start_frames,
                hop_length,
                sample_rate,
                timing_total_frames,
                base_start_frame=base_start_frame,
            )
            if timing_enabled['middle'] else None
        )
        bot_timing = (
            build_timing_schedule_with_offset(
                bot_start_frames,
                hop_length,
                sample_rate,
                timing_total_frames,
                base_start_frame=base_start_frame,
            )
            if timing_enabled['bottom'] else None
        )

        print('--- Full-Length Transformer Test Plan ---')
        print(f'Full-length until: {full_length_until} (target depth={target_depth})')
        print(
            'Sequence mode: one top window; child windows use overlap prefixes '
            f'({full_length_overlap_fraction:.2f} overlap)'
        )
        print(f'Top sequence frames: {top_tf} ({(top_tf * hop_length) / sample_rate:.2f}s)')
        print(f'Timing duration: {(timing_total_frames * hop_length) / sample_rate:.2f}s')
        print(f'Level windows: Top={top_tf}, Middle={mid_tf}, Bottom={bot_tf} frames')
        print(
            f'Chunks needed: Top={len(top_start_frames)}, '
            f'Middle={len(mid_start_frames)}, Bottom={len(bot_start_frames)}'
        )
        if bottom_chunks_per_middle is not None:
            if full_length_overlap_fraction > 0.0:
                print(
                    f'Overlap expansion: {len(mid_start_frames)} middle windows over the top span; '
                    f'{len(bot_start_frames)} bottom windows over the top span'
                )
            else:
                print(
                    f'Hierarchy ratio: {middle_chunks_per_top} middle chunks per top; '
                    f'{bottom_chunks_per_middle} bottom chunks per middle '
                    f'({len(bot_start_frames)} bottom chunks total)'
                )
        print(
            'Timing conditioning: '
            f"top={'enabled' if timing_enabled['top'] else 'disabled'}, "
            f"middle={'enabled' if timing_enabled['middle'] else 'disabled'}, "
            f"bottom={'enabled' if timing_enabled['bottom'] else 'disabled'}"
        )
        if real_top_conditioning is not None:
            print(
                f'Using real top quantized conditioning from {real_top_conditioning["source_path"]} '
                f'(base_start_frame={base_start_frame}, timing_total_frames={timing_total_frames}).'
            )

        generated_by_level = {}

        if real_top_conditioning is not None:
            print('Loading full top-level timeline from real quantized top codes...')
            full_top_tokens = np.asarray(real_top_conditioning['tokens']).astype(np.int64, copy=False)
            top_tokens_list = [full_top_tokens]
        else:
            print('Generating full top-level timeline...')
            top_tokens_list = _generate_level_windows(
                prior=top_prior,
                seq_len=top_seq_len,
                num_samples=num_samples,
                start_frames=top_start_frames,
                device=device,
                temperature=temperature,
                top_k=top_k_value,
                timing_list=top_timing,
                level_name='top',
                progress_interval=progress_interval,
            )
            full_top_tokens = assemble_token_timeline(
                tokens_list=top_tokens_list,
                start_frames=top_start_frames,
                level_time_frames=top_tf,
                level_grid=top_grid,
                total_frames=generated_span_frames,
            ).astype(np.int64, copy=False)
        top_tensor = torch.from_numpy(full_top_tokens)
        top_dynamic_grid = dynamic_grid_for_tokens(top_tensor, top_grid)
        save_indices_with_visualizations(full_top_tokens, save_dir, 'top_full', top_dynamic_grid)
        np.savez(
            os.path.join(save_dir, 'top_window_indices.npz'),
            **{f'window_{idx:04d}': tokens.astype(np.int64, copy=False) for idx, tokens in enumerate(top_tokens_list)}
        )
        generated_by_level['top'] = (top_tensor, top_dynamic_grid, int(top_grid[0]), generated_span_frames)
        print('Assembled full top tokens shape:', full_top_tokens.shape)

        full_middle_tokens = None
        if target_depth >= 1:
            print('Generating full middle-level timeline...')
            middle_model_cfg = middle_config.get('model', {}) if isinstance(middle_config, dict) else {}
            mid_slice_len = _infer_slice_len(
                middle_model_cfg,
                target_seq_len=middle_seq_len,
                inferred_len_key='inferred_cond_seq_len',
                inferred_stride_key='inferred_upsample_stride',
            )
            top_slices_for_middle = [
                get_token_slice_for_frame(
                    full_tokens=full_top_tokens,
                    start_frame=start_frame,
                    level_time_frames=top_tf,
                    level_grid=top_grid,
                    slice_len=mid_slice_len,
                    base_start_frame=0,
                )
                for start_frame in mid_start_frames
            ]
            middle_tokens_list = _generate_level_windows(
                prior=middle_prior,
                seq_len=middle_seq_len,
                num_samples=num_samples,
                start_frames=mid_start_frames,
                device=device,
                temperature=temperature,
                top_k=top_k_value,
                upper_tokens_list=top_slices_for_middle,
                timing_list=mid_timing,
                level_name='middle',
                progress_interval=progress_interval,
                level_time_frames=mid_tf,
                level_grid=middle_grid,
                use_overlap_prefixes=full_length_overlap_fraction > 0.0,
            )
            if full_length_overlap_fraction > 0.0:
                validate_window_prefixes(
                    middle_tokens_list,
                    mid_start_frames,
                    mid_tf,
                    middle_grid,
                    'middle',
                )
            full_middle_tokens = assemble_token_timeline(
                tokens_list=middle_tokens_list,
                start_frames=mid_start_frames,
                level_time_frames=mid_tf,
                level_grid=middle_grid,
                total_frames=generated_span_frames,
            ).astype(np.int64, copy=False)
            middle_tensor = torch.from_numpy(full_middle_tokens)
            middle_dynamic_grid = dynamic_grid_for_tokens(middle_tensor, middle_grid)
            save_indices_with_visualizations(full_middle_tokens, save_dir, 'middle_full', middle_dynamic_grid)
            np.savez(
                os.path.join(save_dir, 'middle_window_indices.npz'),
                **{f'window_{idx:04d}': tokens.astype(np.int64, copy=False) for idx, tokens in enumerate(middle_tokens_list)}
            )
            generated_by_level['middle'] = (middle_tensor, middle_dynamic_grid, int(middle_grid[0]), generated_span_frames)
            print('Assembled full middle tokens shape:', full_middle_tokens.shape)

        if target_depth >= 2:
            print('Generating full bottom-level timeline...')
            if full_middle_tokens is None:
                raise RuntimeError('Cannot generate full bottom timeline without full middle tokens.')

            bottom_model_cfg = bottom_config.get('model', {}) if isinstance(bottom_config, dict) else {}
            bot_mid_slice_len = _infer_slice_len(
                bottom_model_cfg,
                target_seq_len=bottom_seq_len,
                inferred_len_key='inferred_cond_seq_len',
                inferred_stride_key='inferred_upsample_stride',
            )
            middle_slices_for_bottom = [
                get_token_slice_for_frame(
                    full_tokens=full_middle_tokens,
                    start_frame=start_frame,
                    level_time_frames=mid_tf,
                    level_grid=middle_grid,
                    slice_len=bot_mid_slice_len,
                    base_start_frame=0,
                )
                for start_frame in bot_start_frames
            ]

            top_slices_for_bottom = None
            if bottom_condition_on_top:
                bot_top_slice_len = _infer_slice_len(
                    bottom_model_cfg,
                    target_seq_len=bottom_seq_len,
                    inferred_len_key='inferred_second_cond_seq_len',
                    inferred_stride_key='inferred_second_upsample_stride',
                )
                top_slices_for_bottom = [
                    get_token_slice_for_frame(
                        full_tokens=full_top_tokens,
                        start_frame=start_frame,
                        level_time_frames=top_tf,
                        level_grid=top_grid,
                        slice_len=bot_top_slice_len,
                        base_start_frame=0,
                    )
                    for start_frame in bot_start_frames
                ]

            bottom_tokens_list = _generate_level_windows(
                prior=bottom_prior,
                seq_len=bottom_seq_len,
                num_samples=num_samples,
                start_frames=bot_start_frames,
                device=device,
                temperature=temperature,
                top_k=top_k_value,
                upper_tokens_list=middle_slices_for_bottom,
                second_upper_tokens_list=top_slices_for_bottom,
                timing_list=bot_timing,
                level_name='bottom',
                progress_interval=progress_interval,
                level_time_frames=bot_tf,
                level_grid=bottom_grid,
                use_overlap_prefixes=full_length_overlap_fraction > 0.0,
            )
            if full_length_overlap_fraction > 0.0:
                validate_window_prefixes(
                    bottom_tokens_list,
                    bot_start_frames,
                    bot_tf,
                    bottom_grid,
                    'bottom',
                )
            full_bottom_tokens = assemble_token_timeline(
                tokens_list=bottom_tokens_list,
                start_frames=bot_start_frames,
                level_time_frames=bot_tf,
                level_grid=bottom_grid,
                total_frames=generated_span_frames,
            ).astype(np.int64, copy=False)
            bottom_tensor = torch.from_numpy(full_bottom_tokens)
            bottom_dynamic_grid = dynamic_grid_for_tokens(bottom_tensor, bottom_grid)
            save_indices_with_visualizations(full_bottom_tokens, save_dir, 'bottom_full', bottom_dynamic_grid)
            np.savez(
                os.path.join(save_dir, 'bottom_window_indices.npz'),
                **{f'window_{idx:04d}': tokens.astype(np.int64, copy=False) for idx, tokens in enumerate(bottom_tokens_list)}
            )
            generated_by_level['bottom'] = (bottom_tensor, bottom_dynamic_grid, int(bottom_grid[0]), generated_span_frames)
            print('Assembled full bottom tokens shape:', full_bottom_tokens.shape)

        del top_prior, middle_prior, bottom_prior
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        requested_levels = ['top', 'middle', 'bottom'] if decode_level == 'all' else [decode_level]
        levels_to_decode = [level for level in requested_levels if level in generated_by_level]
        skipped_levels = [level for level in requested_levels if level not in generated_by_level]
        if skipped_levels:
            print(
                'Skipping decode for levels not generated by full_length_until='
                f'{full_length_until}: {", ".join(skipped_levels)}'
            )

        for level in levels_to_decode:
            effective_decode_vqvae = _resolve_level_vqvae_path(level, bottom_vqvae_path, vqvae_cfg)
            if not effective_decode_vqvae:
                raise ValueError(
                    f'{level} VQ-VAE path is required. Update vqvae.{level}_model_dir in Transformer config '
                    f'or pass --bottom_vqvae for the bottom level.'
                )
            decode_tokens, decode_grid, chunk_time_cols, trim_frames = generated_by_level[level]
            effective_context_cols = _resolve_decode_context_cols(decode_context_cols, chunk_time_cols)
            _decode_level_to_audio(
                level=level,
                tokens=decode_tokens,
                grid=decode_grid,
                vqvae_path=effective_decode_vqvae,
                weights_file=effective_vqvae_weights_file,
                audio_method=audio_method,
                save_dir=save_dir,
                device=device,
                chunk_time_cols=chunk_time_cols,
                decode_context_cols=effective_context_cols,
                trim_frames=trim_frames,
                min_max_values_path_override=min_max_values_path,
            )
        return
    transformer_dataset_cfg = bottom_config.get('dataset', {}) if isinstance(bottom_config, dict) else {}
    dataset_cfg = dict(transformer_dataset_cfg)
    if effective_bottom_vqvae:
        vqvae_config_path = resolve_vqvae_config_path(effective_bottom_vqvae)
        vqvae_config = load_config(vqvae_config_path)
        if isinstance(vqvae_config, dict):
            dataset_cfg = {**vqvae_config.get('dataset', {}), **dataset_cfg}

    level_target_time_frames = transformer_dataset_cfg.get('level_target_time_frames') or {}
    target_time_frames = int(level_target_time_frames.get('top', transformer_dataset_cfg.get('target_time_frames', 2048)))
    hop_length = int(dataset_cfg.get('hop_length', 256))
    sample_rate = int(dataset_cfg.get('sample_rate', 22050))

    real_top_conditioning = None
    if real_top_quantized_path:
        real_top_conditioning = _load_real_top_conditioning(
            quantized_path=real_top_quantized_path,
            num_samples=num_samples,
            top_time_frames=target_time_frames,
            top_grid=top_grid,
            sample_rate=sample_rate,
            hop_length=hop_length,
            requested_start_frame=real_top_start_frame,
        )

    if timing is None:
        if real_top_conditioning is not None:
            timing = real_top_conditioning['timing'].to(device=device, dtype=torch.float32)
        else:
            timing = torch.zeros((num_samples, 3), dtype=torch.float32, device=device)
            timing[:, 1] = float(timing_duration_seconds)

    depth_by_level = {'top': 0, 'middle': 1, 'bottom': 2, 'all': 2}
    target_depth = depth_by_level[decode_level] if generate_until == 'auto' else depth_by_level[generate_until]
    print('--- Single-Sequence Transformer Test Plan ---')
    print(
        f'Sequence lengths: top={top_seq_len}, middle={middle_seq_len}, bottom={bottom_seq_len} tokens'
    )
    print(f'Generate until: {generate_until} (target depth={target_depth})')
    if bottom_seq_len > 1024 and target_depth >= 2:
        print(
            'Note: this legacy bottom prior generates a long sequence '
            f'({bottom_seq_len} autoregressive steps). Progress will print every '
            f'{progress_interval} tokens.'
        )

    if real_top_conditioning is not None:
        print(
            f'Using real top quantized conditioning from {real_top_conditioning["source_path"]} '
            'instead of sampling from the top prior.'
        )
        top_tokens = torch.from_numpy(np.asarray(real_top_conditioning['tokens']).astype(np.int64, copy=False)).to(device)
    else:
        print(f'Generating top-level indices...')
        with torch.no_grad():
            top_tokens = top_prior.generate(
                batch_size=num_samples,
                start_tokens=None,
                seq_len=top_seq_len,
                temperature=temperature,
                top_k=top_k_value,
                device=device,
                timing=timing,
                progress_label='top',
                progress_interval=progress_interval,
            )

    middle_tokens = None
    if target_depth >= 1:
        print(f'Generating middle-level indices...')
        with torch.no_grad():
            middle_tokens = middle_prior.generate(
                batch_size=num_samples,
                start_tokens=None,
                upper_indices=top_tokens,
                seq_len=middle_seq_len,
                temperature=temperature,
                top_k=top_k_value,
                device=device,
                timing=timing,
                progress_label='middle',
                progress_interval=progress_interval,
            )

    bottom_tokens = None
    if target_depth >= 2:
        print(f'Generating bottom-level indices...')
        with torch.no_grad():
            bottom_tokens = bottom_prior.generate(
                batch_size=num_samples,
                start_tokens=None,
                upper_indices=middle_tokens,
                second_upper_indices=top_tokens if bottom_condition_on_top else None,
                seq_len=bottom_seq_len,
                temperature=temperature,
                top_k=top_k_value,
                device=device,
                timing=timing,
                progress_label='bottom',
                progress_interval=progress_interval,
            )

    save_indices_with_visualizations(top_tokens.cpu().numpy().astype(np.int64), save_dir, 'top', top_grid)
    if middle_tokens is not None:
        save_indices_with_visualizations(middle_tokens.cpu().numpy().astype(np.int64), save_dir, 'middle', middle_grid)
    if bottom_tokens is not None:
        save_indices_with_visualizations(bottom_tokens.cpu().numpy().astype(np.int64), save_dir, 'bottom', bottom_grid)
    del top_prior, middle_prior, bottom_prior
    torch.cuda.empty_cache()

    generated_by_level = {
        'top': (top_tokens, top_grid),
    }
    if middle_tokens is not None:
        generated_by_level['middle'] = (middle_tokens, middle_grid)
    if bottom_tokens is not None:
        generated_by_level['bottom'] = (bottom_tokens, bottom_grid)

    requested_levels = ['top', 'middle', 'bottom'] if decode_level == 'all' else [decode_level]
    levels_to_decode = [level for level in requested_levels if level in generated_by_level]
    skipped_levels = [level for level in requested_levels if level not in generated_by_level]
    if skipped_levels:
        print(
            'Skipping decode for levels not generated by generate_until='
            f'{generate_until}: {", ".join(skipped_levels)}'
        )

    for level in levels_to_decode:
        effective_decode_vqvae = _resolve_level_vqvae_path(level, bottom_vqvae_path, vqvae_cfg)
        if not effective_decode_vqvae:
            raise ValueError(
                f'{level} VQ-VAE path is required. Update vqvae.{level}_model_dir in Transformer config '
                f'or pass --bottom_vqvae for the bottom level.'
            )
        decode_tokens, decode_grid = generated_by_level[level]
        _decode_level_to_audio(
            level=level,
            tokens=decode_tokens,
            grid=decode_grid,
            vqvae_path=effective_decode_vqvae,
            weights_file=effective_vqvae_weights_file,
            audio_method=audio_method,
            save_dir=save_dir,
            device=device,
            min_max_values_path_override=min_max_values_path,
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample top/middle/bottom VQ indices from trained Transformer priors')
    for n in ('top_prior', 'middle_prior', 'bottom_prior'):
        parser.add_argument(
            f'--{n}',
            required=True,
            help=f'Path to Transformer {n} prior run directory, config, or .pth',
        )
    parser.add_argument('--bottom_vqvae', type=str, default=None, help='Path to bottom VQ-VAE run directory, config, or .pth')
    parser.add_argument('--audio_method', type=str, default='griffinlim', help='Audio inversion: griffinlim or istft')
    parser.add_argument('--weights_file', type=str, default='best_model.pth')
    parser.add_argument('--n_samples', type=int, default=6, help='Number of samples to generate (default: 6)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (default: 1.0)')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k filtering for sampling (0 or negative for no filtering)')
    parser.add_argument(
        '--decode_level',
        default='all',
        choices=['all','bottom','middle','top'],
        help ='Which level to decode to audio (default: all)'
    )
    parser.add_argument(
        '--generate_until',
        default='auto',
        choices=['auto','top','middle','bottom'],
        help ='Generate until a certain level (default: auto)'
    )
    parser.add_argument('--full_length', action='store_true', help='Generate full length audio')
    parser.add_argument(
        '--full_length_until',
        default='auto',
        choices=['auto','top','middle','bottom'],
        help ='Generate full length until a certain level (default: auto)'
    )
    parser.add_argument(
        '--full_length_overlap_fraction',
        type=float,
        default=0.5,
        help='Overlap between full-length child windows. 0.5 means 50%% overlap (default: 0.5)'
    )
    parser.add_argument(
        '--timing_duration_seconds',
        type=float,
        default=240.0,
        help='Synthetic song duration for timing conditioning when sampling from the top prior (default: 240s)'
    )
    parser.add_argument(
        '--disable_timing_conditioning',
        action='store_true',
        help='Load timing-capable priors without passing timing conditioning'
    )
    parser.add_argument('--real_top_quantized', help='Path to real top quantized audio')
    parser.add_argument('--real_top_start_frame', type=int,default=0, help='Start frame for real top quantized audio')

    args = parser.parse_args()

    if args.temperature <= 0:
        raise ValueError(f'--temperature must be > 0, got {args.temperature}')
    if args.top_k is not None and args.top_k < 0:
        raise ValueError(f'--top_k must be >= 0, got {args.top_k}')
    if args.full_length_overlap_fraction < 0.0 or args.full_length_overlap_fraction >= 1.0:
        raise ValueError(
            f'--full_length_overlap_fraction must be in [0, 1), got {args.full_length_overlap_fraction}'
        )
    if args.timing_duration_seconds <= 0:
        raise ValueError(f'--timing_duration_seconds must be > 0, got {args.timing_duration_seconds}')

    if args.audio_method not in ('griffinlim', 'istft'):
        raise ValueError("--audio_method must be 'griffinlim' or 'istft'")

    test_transformer_prior(
        top_prior_path=args.top_prior,
        middle_prior_path=args.middle_prior,
        bottom_prior_path=args.bottom_prior,
        bottom_vqvae_path=args.bottom_vqvae,
        audio_method=args.audio_method,
        num_samples=args.n_samples,
        temperature=args.temperature,
        top_k=args.top_k,
        weights_file=args.weights_file,
        decode_level=args.decode_level,
        generate_until=args.generate_until,
        full_length=args.full_length,
        full_length_until=args.full_length_until,
        full_length_overlap_fraction=args.full_length_overlap_fraction,
        timing_duration_seconds=args.timing_duration_seconds,
        disable_timing_conditioning=args.disable_timing_conditioning,
        real_top_quantized_path=args.real_top_quantized,
        real_top_start_frame=args.real_top_start_frame
    )
