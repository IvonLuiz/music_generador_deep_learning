import os
import pickle
import sys
import argparse
import math
import json
from datetime import datetime
import numpy as np
import time

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_config, set_global_seed
from train_scripts.jukebox_utils import load_jukebox_model
from evaluation.transformer_prior import load_transformer_prior
from generation.audio_inversion_cli import add_audio_inversion_args
from generation.generation_config import GenerationConfig
from generation.key_conditioning_cli import add_key_conditioning_args, resolve_key_conditioning
from generation.transformer_io_utils import (
    resolve_min_max_values_path,
    resolve_vqvae_config_path,
    save_decoded_spectrograms,
)
from generation.transformer_sampling_utils import (
    DEFAULT_BOTTOM_RUN_ROOT,
    DEFAULT_MIDDLE_RUN_ROOT,
    DEFAULT_TOP_RUN_ROOT,
    assemble_spectrogram_chunks,
    compute_windowed_step,
    decode_full_level_spectrogram,
    decode_token_blocks,
    generate_level_windows,
    infer_slice_len,
    load_windowed_quantization_config,
    resolve_decode_context_cols,
    resolve_prior_config_path,
    save_audio_from_spectrogram,
)
from windowed_data_utils import (
    assemble_token_timeline,
    build_level_starts,
    build_timing_schedule,
    get_token_slice_for_frame,
    validate_window_prefixes,
)
from processing.preprocess_audio import SAMPLE_RATE, HOP_LENGTH, FRAME_SIZE


def _normalize_min_max_values_path(path: str) -> str:
    candidate = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(candidate):
        candidate = os.path.join(candidate, 'min_max_values.pkl')
    return candidate


def _debug(msg: str) -> None:
    print(f'[DEBUG] {msg}')


def _json_safe_cli_args(args) -> dict:
    """!
    @brief Return argparse values without runtime dataclass objects.
    @param args argparse namespace or compatible object.
    @return JSON-serializable CLI argument dictionary.
    """
    values = dict(vars(args))
    values.pop('generation_config', None)
    values.pop('audio_inversion_config', None)
    return values


def main():
    """!
    @brief CLI entry point for hierarchical music generation.
    """
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
        choices=['windowed', 'independent', 'no_overlap'],
        help='Sampling strategy. windowed reuses overlapping previous codes as context (default); independent uses training hops without copied prefixes; no_overlap uses full-window hops.',
    )
    parser.add_argument(
        '--overlap_fraction',
        type=float,
        default=0.5,
        help='Overlap fraction for windowed sampling (default: 0.5).',
    )
    parser.add_argument(
        '--windowed_prefix_levels',
        type=str,
        default='all',
        choices=['all', 'top'],
        help='Windowed levels. Use top to make middle/bottom use full-window hops without overlap/prefixes.',
    )
    parser.add_argument(
        '--duration_seconds',
        type=float,
        default=30.0,
        help='Target generated song duration in seconds.',
    )
    parser.add_argument(
        '--bottom_decode_mode',
        type=str,
        default='timeline',
        choices=['timeline', 'windowed'],
        help='Decode final bottom spectrogram from the assembled token timeline (default) or legacy window crossfade.',
    )
    parser.add_argument(
        '--bottom_decode_context_cols',
        type=int,
        default=-1,
        help='Extra latent token columns on each side when timeline-decoding bottom chunks. Use -1 for half-window context.',
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible generation (set to negative to disable)')
    add_audio_inversion_args(parser, include_denormalization=True)
    add_key_conditioning_args(parser)
    parser.add_argument(
        '--save_middle_audio',
        action='store_true',
        default=True,
        help='Also invert the decoded full middle spectrogram and save it as audio/middle_sample.wav.',
    )
    parser.add_argument('--save_root', type=str, default='samples/generate_music_maestro', help='Root directory for generated outputs')
    args = parser.parse_args()
    resolve_key_conditioning(args.key, args.key_id)

    generation_config = GenerationConfig.from_args(args)
    generation_config.apply_to_args(args)
    set_global_seed(args.seed, deterministic=True)
    if args.seed is not None:
        print(f'Using deterministic seed: {args.seed}')

    try:
        save_dir = generate_hierarchical_music(args)
        print(f'Done! Saved generated samples to {save_dir}')
    except Exception as e:
        _debug(f'Generation failed in main with error: {type(e).__name__}: {e}')
        raise
def generate_hierarchical_music(args) -> str:
    """!
    @brief Generate, decode, invert, and save one hierarchical music sample.
    @param args argparse namespace or compatible object with generation fields.
    @return Output run directory.
    """
    generation_config = getattr(args, 'generation_config', GenerationConfig.from_args(args))
    generation_config.apply_to_args(args)
    audio_config = generation_config.audio

    _debug('Resolving config paths...')
    top_transformer_prior_config_path = resolve_prior_config_path(args.top_config, args.top_run_root, 'top')
    middle_transformer_prior_config_path = resolve_prior_config_path(args.middle_config, args.middle_run_root, 'middle')
    bottom_transformer_prior_config_path = resolve_prior_config_path(args.bottom_config, args.bottom_run_root, 'bottom')

    print(f'Top config: {top_transformer_prior_config_path}')
    print(f'Middle config: {middle_transformer_prior_config_path}')
    print(f'Bottom config: {bottom_transformer_prior_config_path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _debug(f'Using device: {device}')

    # Load the three trained priors.
    _debug('Loading transformer priors...')
    top_prior, top_config, top_prior_model_path = load_transformer_prior('top', top_transformer_prior_config_path, device, weights_file=args.weights_file)
    middle_prior, middle_config, middle_prior_model_path = load_transformer_prior('middle', middle_transformer_prior_config_path, device, weights_file=args.weights_file)
    bottom_prior, bottom_config, bottom_prior_model_path = load_transformer_prior('bottom', bottom_transformer_prior_config_path, device, weights_file=args.weights_file)
    key_config = resolve_key_conditioning(getattr(args, 'key', None), getattr(args, 'key_id', None))
    key_ids = None
    key_conditioning_metadata = None
    key_enabled = {
        'top': bool(getattr(top_prior, 'use_key_conditioning', False)),
        'middle': bool(getattr(middle_prior, 'use_key_conditioning', False)),
        'bottom': bool(getattr(bottom_prior, 'use_key_conditioning', False)),
    }
    if key_config is not None:
        key_ids = torch.tensor([int(key_config.key_id)], dtype=torch.long, device=device)
        key_conditioning_metadata = key_config.to_dict()
        print(f'Key conditioning requested: {key_config.key_label} (id={key_config.key_id})')
        print(
            'Key conditioning support: '
            f"top={'enabled' if key_enabled['top'] else 'disabled'}, "
            f"middle={'enabled' if key_enabled['middle'] else 'disabled'}, "
            f"bottom={'enabled' if key_enabled['bottom'] else 'disabled'}"
        )
        if not any(key_enabled.values()):
            print('Warning: key conditioning was requested, but these Transformer priors do not use key embeddings.')
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
    bottom_decode_context_cols = resolve_decode_context_cols(
        args.bottom_decode_context_cols,
        int(bottom_grid[0]),
    )

    dataset_cfg = bottom_config.get('dataset', {}) if isinstance(bottom_config, dict) else {}
    quantization_cfg, quantized_path = load_windowed_quantization_config(bottom_config)
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
        top_step, top_overlap, top_overlap_cols, top_hop_cols = compute_windowed_step(top_tf, top_grid, args.overlap_fraction)
        mid_step, mid_overlap, mid_overlap_cols, mid_hop_cols = compute_windowed_step(mid_tf, middle_grid, args.overlap_fraction)
        bot_step, bot_overlap, bot_overlap_cols, bot_hop_cols = compute_windowed_step(bot_tf, bottom_grid, args.overlap_fraction)
        if args.windowed_prefix_levels == 'top':
            mid_step = mid_tf
            bot_step = bot_tf
            mid_overlap = 0.0
            bot_overlap = 0.0
            mid_overlap_cols = 0
            bot_overlap_cols = 0
            mid_hop_cols = int(middle_grid[0])
            bot_hop_cols = int(bottom_grid[0])
        effective_overlap = {'top': top_overlap, 'middle': mid_overlap, 'bottom': bot_overlap}
        overlap_cols = {'top': top_overlap_cols, 'middle': mid_overlap_cols, 'bottom': bot_overlap_cols}
        hop_cols = {'top': top_hop_cols, 'middle': mid_hop_cols, 'bottom': bot_hop_cols}
    elif args.sampling_mode == 'no_overlap':
        top_step, mid_step, bot_step = top_tf, mid_tf, bot_tf
        effective_overlap = {'top': 0.0, 'middle': 0.0, 'bottom': 0.0}
        overlap_cols = {'top': 0, 'middle': 0, 'bottom': 0}
        hop_cols = {
            'top': int(top_grid[0]),
            'middle': int(middle_grid[0]),
            'bottom': int(bottom_grid[0]),
        }
    else:
        top_step, mid_step, bot_step = training_top_step, training_mid_step, training_bot_step

    min_max_values_path = None
    if not audio_config.use_fixed_db_scale:
        _debug('Resolving min_max_values.pkl path...')
        if audio_config.min_max_values_path:
            min_max_values_path = _normalize_min_max_values_path(audio_config.min_max_values_path)
            _debug(f'Using explicitly provided min_max_values_path: {min_max_values_path}')
        else:
            try:
                min_max_values_path = resolve_min_max_values_path(bottom_config, debug_fn=_debug)
            except FileNotFoundError as e:
                print(f'Warning: {str(e)} Proceeding without it.')
    else:
        _debug('Skipping min_max_values.pkl resolution (using fixed dB scale)')

    if min_max_values_path is None and not audio_config.use_fixed_db_scale:
        print('Warning: min_max_values_path not found, falling back to fixed_db_scale.')
        audio_config.use_fixed_db_scale = True

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
    mid_slice_len = infer_slice_len(
        middle_model_cfg,
        target_seq_len=middle_seq_len,
        inferred_len_key='inferred_cond_seq_len',
        inferred_stride_key='inferred_upsample_stride',
    )
    bot_mid_slice_len = infer_slice_len(
        bottom_model_cfg,
        target_seq_len=bottom_seq_len,
        inferred_len_key='inferred_cond_seq_len',
        inferred_stride_key='inferred_upsample_stride',
    )
    bot_top_slice_len = 0
    if bottom_condition_on_top:
        bot_top_slice_len = infer_slice_len(
            bottom_model_cfg,
            target_seq_len=bottom_seq_len,
            inferred_len_key='inferred_second_cond_seq_len',
            inferred_stride_key='inferred_second_upsample_stride',
        )
    
    base_start_frame = 0
    total_source_frames = max(1, int(math.ceil((args.duration_seconds * sample_rate) / hop_length)))
    use_top_windowed_prefix = args.sampling_mode == 'windowed'
    use_child_windowed_prefix = args.sampling_mode == 'windowed' and args.windowed_prefix_levels == 'all'

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
        print(f"Windowed prefix levels: {args.windowed_prefix_levels}")
        print(
            "Requested/effective overlap: "
            f"{args.overlap_fraction:.2f} / "
            f"top={effective_overlap['top']:.3f}, middle={effective_overlap['middle']:.3f}, bottom={effective_overlap['bottom']:.3f}"
        )
    print(f"Requested generated duration: {args.duration_seconds:.2f}s")
    print(
        "Sampling controls: "
        f"temp={args.temperature:.3f}, top_k={args.top_k}"
    )
    print(f"Timing total duration: {total_source_duration_s:.2f}s ({total_source_frames} frames)")
    print(f"Sampling window steps: Top={top_step}, Mid={mid_step}, Bot={bot_step} frames")
    print(f"Training window steps: Top={training_top_step}, Mid={training_mid_step}, Bot={training_bot_step} frames")
    timing_enabled = {
        'top': top_prior.use_timing_conditioning,
        'middle': middle_prior.use_timing_conditioning,
        'bottom': bottom_prior.use_timing_conditioning,
    }
    print(
        "Timing conditioning during generation: "
        f"top={'enabled' if timing_enabled['top'] else 'disabled'}, "
        f"middle={'enabled' if timing_enabled['middle'] else 'disabled'}, "
        f"bottom={'enabled' if timing_enabled['bottom'] else 'disabled'}"
    )
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

    top_timing = (
        build_timing_schedule(top_start_frames, hop_length, sample_rate, total_source_frames)
        if timing_enabled['top'] else None
    )
    mid_timing = (
        build_timing_schedule(mid_start_frames, hop_length, sample_rate, total_source_frames)
        if timing_enabled['middle'] else None
    )
    bot_timing = (
        build_timing_schedule(bot_start_frames, hop_length, sample_rate, total_source_frames)
        if timing_enabled['bottom'] else None
    )

    # Step 1: Top-Level Unrolling (The Composer)
    ## Generate global structure block-by-block, reusing previous overlap in windowed mode.
    start_time = time.time()
    top_tokens_list = generate_level_windows(
        prior=top_prior,
        seq_len=top_seq_len,
        num_samples=1,
        start_frames=top_start_frames,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        upper_tokens_list=None,
        timing_list=top_timing,
        key_ids=key_ids,
        level_name='top',
        level_time_frames=top_tf,
        level_grid=top_grid,
        use_overlap_prefixes=use_top_windowed_prefix,
    )
    if use_top_windowed_prefix:
        validate_window_prefixes(top_tokens_list, top_start_frames, top_tf, top_grid, 'top')
    full_top_tokens = assemble_token_timeline(
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
    middle_tokens_list = generate_level_windows(
        prior=middle_prior,
        seq_len=middle_seq_len,
        num_samples=1,
        start_frames=mid_start_frames,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        upper_tokens_list=top_slices_for_middle,
        timing_list=mid_timing,
        key_ids=key_ids,
        level_name='middle',
        level_time_frames=mid_tf,
        level_grid=middle_grid,
        use_overlap_prefixes=use_child_windowed_prefix,
    )
    if use_child_windowed_prefix:
        validate_window_prefixes(middle_tokens_list, mid_start_frames, mid_tf, middle_grid, 'middle')

    full_middle_tokens = assemble_token_timeline(
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
    bottom_tokens_list = generate_level_windows(
        prior=bottom_prior,
        seq_len=bottom_seq_len,
        num_samples=1,
        start_frames=bot_start_frames,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        upper_tokens_list=middle_slices_for_bottom,
        second_upper_tokens_list=top_slices_for_bottom,
        timing_list=bot_timing,
        key_ids=key_ids,
        level_name='bottom',
        level_time_frames=bot_tf,
        level_grid=bottom_grid,
        use_overlap_prefixes=use_child_windowed_prefix,
    )
    if use_child_windowed_prefix:
        validate_window_prefixes(bottom_tokens_list, bot_start_frames, bot_tf, bottom_grid, 'bottom')

    full_bottom_tokens = assemble_token_timeline(
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
    if all(timing_enabled.values()):
        timing_source = 'duration_conditioned_window_schedule'
    elif not any(timing_enabled.values()):
        timing_source = 'disabled'
    else:
        timing_source = 'partial_duration_conditioned_window_schedule'
    timing_metadata = {
        'timing_source': timing_source,
        'timing_conditioning_enabled': dict(timing_enabled),
        'key_conditioning': key_conditioning_metadata,
        'key_conditioning_enabled': dict(key_enabled),
        'sampling_mode': args.sampling_mode,
        'windowed_prefix_levels': args.windowed_prefix_levels,
        'spectrogram_assembly': 'full_token_timeline' if args.bottom_decode_mode == 'timeline' else 'linear_crossfade',
        'bottom_decode_mode': args.bottom_decode_mode,
        'bottom_decode_context_cols': int(bottom_decode_context_cols),
        'requested_bottom_decode_context_cols': int(args.bottom_decode_context_cols),
        'sampling_temperature': float(args.temperature),
        'sampling_top_k': args.top_k,
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

    generation_parameters = {
        'generated_at': current_time,
        'command': sys.argv,
        'cli_args': _json_safe_cli_args(args),
        'device': str(device),
        'transformer_priors': {
            'top': {
                'config_path': os.path.abspath(top_transformer_prior_config_path),
                'weights_path': os.path.abspath(top_prior_model_path),
            },
            'middle': {
                'config_path': os.path.abspath(middle_transformer_prior_config_path),
                'weights_path': os.path.abspath(middle_prior_model_path),
            },
            'bottom': {
                'config_path': os.path.abspath(bottom_transformer_prior_config_path),
                'weights_path': os.path.abspath(bottom_prior_model_path),
            },
        },
        'vqvae': {
            'top_model_dir': top_vqvae_ref,
            'middle_model_dir': middle_vqvae_ref,
            'bottom_model_dir': bottom_vqvae_ref,
            'bottom_config_path': os.path.abspath(bottom_vqvae_config_path),
            'weights_file': vqvae_weights_file,
        },
        'min_max_values_path': min_max_values_path,
        'generation_config': generation_config.to_dict(),
        'quantization_config': quantization_cfg,
        'timing_metadata_file': 'generation_timing_metadata.json',
        'indices_dir': 'indices',
        'resolved_generation_settings': {
            'temperature': float(args.temperature),
            'top_k': args.top_k,
            'duration_seconds': float(args.duration_seconds),
            'sampling_mode': args.sampling_mode,
            'windowed_prefix_levels': args.windowed_prefix_levels,
            'requested_overlap_fraction': float(args.overlap_fraction),
            'effective_overlap_fraction': effective_overlap,
            'bottom_decode_mode': args.bottom_decode_mode,
            'bottom_decode_context_cols': int(bottom_decode_context_cols),
            'audio_inversion': audio_config.to_dict(),
            'key_conditioning': key_conditioning_metadata,
            'seed': args.seed,
        },
        'resolved_audio_settings': {
            'sample_rate': int(sample_rate),
            'hop_length': int(hop_length),
            'frame_size': int(frame_size),
            'spectrogram_type': spectrogram_type,
            'n_mels': int(n_mels),
        },
        'resolved_window_settings': {
            'level_time_frames': {'top': int(top_tf), 'middle': int(mid_tf), 'bottom': int(bot_tf)},
            'sampling_step_frames': {'top': int(top_step), 'middle': int(mid_step), 'bottom': int(bot_step)},
            'training_step_frames': {'top': int(training_top_step), 'middle': int(training_mid_step), 'bottom': int(training_bot_step)},
            'chunk_counts': {'top': int(top_chunks), 'middle': int(middle_chunks), 'bottom': int(bottom_chunks)},
            'chunk_start_frames': {
                'top': [int(x) for x in top_start_frames],
                'middle': [int(x) for x in mid_start_frames],
                'bottom': [int(x) for x in bot_start_frames],
            },
        },
    }
    with open(os.path.join(save_dir, 'generation_parameters.json'), 'w', encoding='utf-8') as f:
        json.dump(generation_parameters, f, indent=2)
    print(f"Saved generation parameters to {os.path.join(save_dir, 'generation_parameters.json')}")

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

    middle_decoded_specs = None

    if top_vqvae_ref:
        decode_full_level_spectrogram(
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
        middle_decoded_specs = decode_full_level_spectrogram(
            level='middle',
            vqvae_ref=middle_vqvae_ref,
            full_tokens=full_middle_tokens,
            level_grid=middle_grid,
            total_frames=total_source_frames,
            device=device,
            weights_file=vqvae_weights_file,
            save_dir=save_dir,
        )

    decode_start_time = time.time()
    if args.bottom_decode_mode == 'timeline':
        # Step 3: Spectrogram Assembly (Audio Engineer)
        ## Decode the de-overlapped bottom token timeline once, matching top/middle handling.
        print('Decoding assembled bottom token timeline into spectrograms...')
        final_spectrogram = decode_full_level_spectrogram(
            level='bottom',
            vqvae_ref=bottom_vqvae_ref,
            full_tokens=full_bottom_tokens,
            level_grid=bottom_grid,
            total_frames=total_source_frames,
            device=device,
            weights_file=vqvae_weights_file,
            save_dir=save_dir,
            context_cols=bottom_decode_context_cols,
        )
    else:
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
        ## Legacy path: decode generated windows independently and crossfade spectrograms.
        reconstructed_spectrograms = decode_token_blocks(
            vqvae=vqvae_bottom_decoder,
            tokens_list=bottom_tokens_list,
            level_grid=bottom_grid,
            device=device,
        )
        print('Decoded spectrograms for all blocks. Each block has shape:', reconstructed_spectrograms[0].shape if reconstructed_spectrograms else None)
        final_spectrogram = assemble_spectrogram_chunks(
            spec_chunks=reconstructed_spectrograms,
            start_frames=bot_start_frames,
            total_frames=total_source_frames,
        )

    print('Decoding took: {:.2f} seconds'.format(time.time() - decode_start_time))
    print('Spectrogram reconstruction complete. Final spectrogram shape:', final_spectrogram.shape)

    # Step 4: Spectrogram Inversion (The Mastering Engineer)
    ## convert final spectrograms back to audio and save
    min_max_values = None
    if min_max_values_path and os.path.exists(min_max_values_path):
        _debug(f'Loading min/max values from: {min_max_values_path}')
        with open(min_max_values_path, 'rb') as f:
            min_max_values = pickle.load(f)
    elif not audio_config.use_fixed_db_scale:
        print('Warning: min_max_values_path not found, falling back to fixed_db_scale.')
        audio_config.use_fixed_db_scale = True

    save_decoded_spectrograms(final_spectrogram, save_dir)
    bottom_audio_path = save_audio_from_spectrogram(
        spectrograms=final_spectrogram,
        min_max_values=min_max_values,
        save_dir=save_dir,
        filename='sample.wav',
        hop_length=hop_length,
        sample_rate=sample_rate,
        frame_size=frame_size,
        spectrogram_type=spectrogram_type,
        n_mels=n_mels,
        audio_config=audio_config,
    )
    print(f'Saved bottom audio to {bottom_audio_path}')

    if args.save_middle_audio:
        if middle_decoded_specs is None:
            print('Skipping middle audio because the middle VQ-VAE reference is unavailable.')
        else:
            middle_audio_path = save_audio_from_spectrogram(
                spectrograms=middle_decoded_specs,
                min_max_values=min_max_values,
                save_dir=save_dir,
                filename='middle_sample.wav',
                hop_length=hop_length,
                sample_rate=sample_rate,
                frame_size=frame_size,
                spectrogram_type=spectrogram_type,
                n_mels=n_mels,
                audio_config=audio_config,
            )
            print(f'Saved middle audio to {middle_audio_path}')

    return save_dir


if __name__ == '__main__':
    main()
