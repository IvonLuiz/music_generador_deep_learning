import os
import sys
import yaml
import argparse
import json
import numpy as np
from datetime import datetime
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.jukebox_precomputed_hierarchical_dataset import JukeboxQuantizedDataset
from modeling.torch.transformer_prior_conditioned import TransformerPriorConditioned
from utils import set_global_seed, list_npy_files, load_config
from callbacks import EarlyStopping
from train_scripts.jukebox_utils import parse_level, split_train_val_paths
from train_scripts.resume_utils import load_resume_artifacts

LEVEL_TO_PRIOR_CFG = {'top': 'top_prior', 'middle': 'middle_prior', 'bottom': 'bottom_prior'}
COND_LEVEL = {'top': None, 'middle': 'top', 'bottom': 'middle'}
SECOND_COND_LEVEL = {'top': None, 'middle': None, 'bottom': 'top'}


def plot_losses(train_losses, val_losses, save_dir, best_epoch=None, best_val_loss=None, level_name: str = 'top'):
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss') if val_losses else None

    if best_epoch is not None and best_val_loss is not None:
        plt.scatter(best_epoch, best_val_loss, c='red', marker='*', s=120, label=f'Best (Loss: {best_val_loss:.4f})', zorder=5)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Transformer {level_name.capitalize()} Prior Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()


def _get_prior_cfg(config: dict, name: str) -> dict:
    priors = config.get('priors')
    if priors and name in priors:
        return priors[name]
    return config[name]


def _compute_stride(lower_len: int, upper_len: int, level_name: str) -> int:
    if upper_len <= 0:
        raise ValueError(f'Invalid upper sequence length for {level_name}: {upper_len}')
    if lower_len % upper_len != 0:
        raise ValueError(
            f'Cannot infer upsample_stride for {level_name}: lower_len={lower_len}, upper_len={upper_len}'
        )
    return lower_len // upper_len


def _tensor_grid_shape(name: str, tensor: torch.Tensor) -> Tuple[int, int]:
    if tensor is None or tensor.numel() == 0:
        raise ValueError(f'{name} is required to infer 2D conditioner geometry.')
    if tensor.ndim != 2:
        raise ValueError(f'{name} must have shape (time_cols, freq_bins), got {tuple(tensor.shape)}')
    time_cols, freq_bins = int(tensor.shape[0]), int(tensor.shape[1])
    if time_cols <= 0 or freq_bins <= 0:
        raise ValueError(f'{name} has invalid shape {tuple(tensor.shape)}')
    return time_cols, freq_bins


def _compute_2d_stride_from_tensors(
    target_indices: torch.Tensor,
    cond_indices: torch.Tensor,
    label: str,
) -> Tuple[Tuple[int, int], int]:
    target_time, target_freq = _tensor_grid_shape(f'{label} target', target_indices)
    cond_time, cond_freq = _tensor_grid_shape(f'{label} conditioning', cond_indices)
    if target_time % cond_time != 0 or target_freq % cond_freq != 0:
        raise ValueError(
            f'Cannot infer 2D conditioner stride for {label}: '
            f'target_shape=({target_time}, {target_freq}), '
            f'cond_shape=({cond_time}, {cond_freq})'
        )
    return (target_time // cond_time, target_freq // cond_freq), cond_freq


def _serialize_stride(stride):
    if isinstance(stride, (tuple, list)):
        if len(stride) != 2:
            raise ValueError(f'Expected stride tuple/list of length 2, got {stride}')
        return [int(stride[0]), int(stride[1])]
    return int(stride)


def _load_model_state_compatibly(model: nn.Module, state_dict: dict) -> bool:
    try:
        model.load_state_dict(state_dict)
        return False
    except RuntimeError as exc:
        print(f"Strict checkpoint load failed; retrying with shape-compatible weights only: {exc}")

    current_state = model.state_dict()
    compatible_state = {}
    skipped = []
    unexpected = []
    for key, value in state_dict.items():
        if key not in current_state:
            unexpected.append(key)
            continue
        if tuple(current_state[key].shape) != tuple(value.shape):
            skipped.append(key)
            continue
        compatible_state[key] = value

    missing, unexpected_from_load = model.load_state_dict(compatible_state, strict=False)
    unexpected.extend(unexpected_from_load)
    if skipped:
        print(
            "Skipped checkpoint weights with incompatible shapes "
            f"({len(skipped)} total): {skipped[:10]}"
        )
    if missing:
        print(f"Model parameters initialized fresh ({len(missing)} total): {missing[:10]}")
    if unexpected:
        print(f"Ignored unexpected checkpoint keys ({len(unexpected)} total): {unexpected[:10]}")
    return True


def _loader_kwargs(num_workers: int, pin_memory: bool, persist_workers: bool, prefetch_factor: Optional[int]) -> dict:
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = persist_workers
        if prefetch_factor is not None:
            kwargs['prefetch_factor'] = prefetch_factor
    return kwargs


def train_transformer_prior(
    config_path: str,
    level_override: Optional[str] = None,
):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')

    dataset_cfg = config['dataset']
    vqvae_cfg = config['vqvae']
    transformer_cfg = config.setdefault('model', {})
    selected_level = parse_level(level_override or transformer_cfg.get('selected_level', 'top'))
    train_general_cfg = config['training']
    train_prior_cfg = config['training'][LEVEL_TO_PRIOR_CFG[selected_level]]
    train_cfg = {**train_general_cfg, **train_prior_cfg}  # Prior-specific settings override general settings
    transformer_cfg['selected_level'] = selected_level
    seed = int(train_cfg.get('seed', 42))
    set_global_seed(seed)
    num_workers = int(train_cfg.get('num_workers', 4))
    pin_memory = bool(train_cfg.get('pin_memory', True))
    persist_workers_cfg = train_cfg.get('persist_workers', True)
    persist_workers = bool(persist_workers_cfg) if persist_workers_cfg is not None else True
    prefetch_factor_cfg = train_cfg.get('prefetch_factor', 4)
    prefetch_factor = int(prefetch_factor_cfg) if prefetch_factor_cfg is not None else None

    target_time_frames = int(dataset_cfg.get('target_time_frames', 2048))
    level_target_time_frames = dataset_cfg.get('level_target_time_frames') or {}
    quantized_data_path = dataset_cfg.get('quantized_data_path', './data/processed/maestro_quantized/')
    sample_rate = int(dataset_cfg.get('sample_rate', 22050))
    hop_length = int(dataset_cfg.get('hop_length', 256))
    segment_overlap = float(dataset_cfg.get('segment_overlap', 0.0))

    # Codebook size: use explicit config value or sensible default (Jukebox uses 2048)
    vqvae_codebook_size = int(vqvae_cfg.get('codebook_size', 2048))

    all_file_paths = list_npy_files(dataset_cfg['processed_path'])
    if len(all_file_paths) == 0:
        raise ValueError(f"No .npy files found under dataset path: {dataset_cfg['processed_path']}")
    print(f'Found {len(all_file_paths)} spectrogram files for quantization.')
    print(f'Level target time frames: {level_target_time_frames}')

    validation_split = float(train_cfg.get('validation_split', 0.1))
    train_file_paths, val_file_paths = split_train_val_paths(
        all_file_paths=all_file_paths,
        dataset_cfg=dataset_cfg,
        validation_split=validation_split,
        seed=seed,
    )
    print(
        f"Using {len(train_file_paths)} training files and "
        f"{len(val_file_paths) if val_file_paths else 0} validation files "
        f"(validation_split={validation_split:.3f})."
    )

    # Load precomputed quantized dataset
    print(f"--- Loading Precomputed Quantized Dataset for {selected_level} ---")
    train_dataset = JukeboxQuantizedDataset(
        quantized_path=quantized_data_path,
        file_paths=train_file_paths,
        target_time_frames=target_time_frames,
        level_target_time_frames=level_target_time_frames,
        selected_level=selected_level,
    )
    print(
        f"Train dataset examples for {selected_level}: {len(train_dataset)} "
        f"(loader workers={num_workers}, pin_memory={pin_memory}, "
        f"persistent_workers={persist_workers if num_workers > 0 else False}, prefetch_factor={prefetch_factor})"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        **_loader_kwargs(num_workers, pin_memory, persist_workers, prefetch_factor),
    )
    val_loader = None
    if val_file_paths:
        print(f"--- Loading Precomputed Validation Dataset for {selected_level} ---")
        val_dataset = JukeboxQuantizedDataset(
            quantized_path=quantized_data_path,
            file_paths=val_file_paths,
            target_time_frames=target_time_frames,
            level_target_time_frames=level_target_time_frames,
            selected_level=selected_level,
        )
        print(f"Validation dataset examples for {selected_level}: {len(val_dataset)}")
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg['batch_size'],
            shuffle=False,
            **_loader_kwargs(num_workers, pin_memory, persist_workers, prefetch_factor),
        )
    else:
        print(f"No validation dataset configured for {selected_level}; training will use {len(train_dataset)} examples only.")

    # Use explicit codebook size (not inferred from data, which may have smaller max token)
    num_embeddings_map = {
        'top': vqvae_codebook_size,
        'middle': vqvae_codebook_size,
        'bottom': vqvae_codebook_size,
    }
    print(f"Using codebook size: {vqvae_codebook_size}")

    sample = train_dataset[0]
    # Dataset returns fixed-size window tensors; use those shapes directly rather than
    # inferring conditioner lengths from full-song latent grids.
    target_indices, cond_indices_sample, second_cond_indices_sample, timing_sample = sample

    # Infer grid shapes from the dataset's reported grid info
    top_rows, top_cols = train_dataset.top_grid
    middle_rows, middle_cols = train_dataset.middle_grid
    bottom_rows, bottom_cols = train_dataset.bottom_grid

    grid_shapes = {
        'top': (top_rows, top_cols),
        'middle': (middle_rows, middle_cols),
        'bottom': (bottom_rows, bottom_cols)
    }

    target_seq_len = int(target_indices.numel())
    seq_lens = {
        'top': int(top_rows * top_cols),
        'middle': int(middle_rows * middle_cols),
        'bottom': int(bottom_rows * bottom_cols),
    }
    expected_target_seq_len = seq_lens[selected_level]
    if target_seq_len != expected_target_seq_len:
        raise ValueError(
            f'{selected_level} target has {target_seq_len} tokens, but dataset grid '
            f'{grid_shapes[selected_level] if "grid_shapes" in locals() else "unknown"} implies '
            f'{expected_target_seq_len}.'
        )

    print(f"Top grid shape: ({top_rows}, {top_cols})")
    print(f"Middle grid shape: ({middle_rows}, {middle_cols})")
    print(f"Bottom grid shape: ({bottom_rows}, {bottom_cols})")

    # --- Temporal alignment for conditioning ---
    # Each upsampler trains on a window of the target level, conditioned on the
    # matching ALIGNED slice of the upper level that covers the same audio span.
    prior_cfg = _get_prior_cfg(config, LEVEL_TO_PRIOR_CFG[selected_level])
    cond_level = COND_LEVEL[selected_level]
    second_cond_level = SECOND_COND_LEVEL[selected_level]
    condition_on_top = bool(prior_cfg.get('condition_on_top', False)) and selected_level == 'bottom'
    is_upsampler = prior_cfg.get('is_upsampler', selected_level in ['middle', 'bottom'])

    _top_tf = int(level_target_time_frames.get('top', target_time_frames))
    _mid_tf = int(level_target_time_frames.get('middle', target_time_frames))
    _bot_tf = int(level_target_time_frames.get('bottom', target_time_frames))
    selected_tf = {'top': _top_tf, 'middle': _mid_tf, 'bottom': _bot_tf}[selected_level]
    timing_window_seconds = (selected_tf * hop_length) / sample_rate

    cond_seq_len = int(cond_indices_sample.numel()) if cond_indices_sample.numel() > 0 else 0
    second_cond_seq_len = int(second_cond_indices_sample.numel()) if second_cond_indices_sample.numel() > 0 else 0
    cond_time_cols = int(cond_indices_sample.shape[0]) if cond_indices_sample.ndim == 2 and cond_indices_sample.numel() > 0 else 0
    cond_freq_bins = int(cond_indices_sample.shape[1]) if cond_indices_sample.ndim == 2 and cond_indices_sample.numel() > 0 else 0
    second_cond_time_cols = (
        int(second_cond_indices_sample.shape[0])
        if second_cond_indices_sample.ndim == 2 and second_cond_indices_sample.numel() > 0
        else 0
    )
    second_cond_freq_bins = (
        int(second_cond_indices_sample.shape[1])
        if second_cond_indices_sample.ndim == 2 and second_cond_indices_sample.numel() > 0
        else 0
    )

    if cond_seq_len > 0:
        print(f"Conditioning slice for {selected_level}: {cond_time_cols} time-cols × {cond_freq_bins} freq = {cond_seq_len} tokens")
    if second_cond_seq_len > 0:
        print(f"Second conditioning slice (Top→Bottom): {second_cond_time_cols} time-cols × {second_cond_freq_bins} freq = {second_cond_seq_len} tokens")

    is_upsampler = cond_level is not None
    upsample_stride = None
    cond_num_embeddings = None
    cond_block_len = None
    second_upsample_stride = None
    second_cond_num_embeddings = None
    second_cond_block_len = None
    if is_upsampler:
        cond_num_embeddings = num_embeddings_map[cond_level]
        upsample_stride, cond_block_len = _compute_2d_stride_from_tensors(
            target_indices,
            cond_indices_sample,
            f'{cond_level}->{selected_level}',
        )
        print(f"Primary 2D conditioner stride for {cond_level}->{selected_level}: {upsample_stride}")
    if condition_on_top and second_cond_level is not None:
        second_cond_num_embeddings = num_embeddings_map[second_cond_level]
        second_upsample_stride, second_cond_block_len = _compute_2d_stride_from_tensors(
            target_indices,
            second_cond_indices_sample,
            f'{second_cond_level}->{selected_level}',
        )
        print(f"Second 2D conditioner stride for {second_cond_level}->{selected_level}: {second_upsample_stride}")

    max_time_steps = int(prior_cfg.get('max_time_steps', 500))

    prior = TransformerPriorConditioned(
        num_embeddings=num_embeddings_map[selected_level],
        model_dim=int(prior_cfg['model_dim']),
        num_heads=int(prior_cfg['num_heads']),
        num_layers=int(prior_cfg['num_layers']),
        dim_feedforward=int(prior_cfg['dim_feedforward']),
        max_seq_len=target_seq_len,
        block_len=int(prior_cfg.get('block_len', 16)),
        max_time_steps=max_time_steps,
        is_upsampler=is_upsampler,
        cond_num_embeddings=cond_num_embeddings,
        cond_block_len=cond_block_len,
        upsample_stride=upsample_stride,
        second_cond_num_embeddings=second_cond_num_embeddings,
        second_cond_block_len=second_cond_block_len,
        second_upsample_stride=second_upsample_stride,
        conditioner_residual_block_width=int(prior_cfg.get('conditioner_residual_block_width', 1024)),
        conditioner_residual_blocks=int(prior_cfg.get('conditioner_residual_blocks', 16)),
        conditioner_kernel_size=int(prior_cfg.get('conditioner_kernel_size', 3)),
        conditioner_conv_channels=int(prior_cfg.get('conditioner_conv_channels', 1024)),
        conditioner_dilation_growth_rate=int(prior_cfg.get('conditioner_dilation_growth_rate', 3)),
        conditioner_dilation_cycle=int(prior_cfg.get('conditioner_dilation_cycle', 8)),
        dropout=float(prior_cfg.get('dropout', 0.1)),
        attention_qkv_ratio=float(prior_cfg.get('attention_qkv_ratio', 1.0)),
        use_bos_token=bool(prior_cfg.get('use_bos_token', False)),
        use_timing_conditioning=bool(prior_cfg.get('use_timing_conditioning', True)),
        timing_num_bins=int(prior_cfg.get('timing_num_bins', 1024)),
        duration_num_bins=int(prior_cfg.get('duration_num_bins', 256)),
        timing_window_seconds=float(prior_cfg.get('timing_window_seconds', timing_window_seconds)),
        timing_max_duration_seconds=float(prior_cfg.get('timing_max_duration_seconds', 3600.0)),
        timing_embedding_init_std=float(prior_cfg.get('timing_embedding_init_std', 0.02)),
        timing_embedding_scale=float(prior_cfg.get('timing_embedding_scale', 1.0)),
        use_2d_conditioner=bool(prior_cfg.get('use_2d_conditioner', True)),
    ).to(device)
    print(
        f"Timing conditioning: {'enabled' if prior.use_timing_conditioning else 'disabled'} "
        f"(learned absolute/relative/duration embeddings)"
    )

    retrain = bool(train_cfg.get('retrain', False))
    pretrained_weights_path = train_cfg.get('pretrained_weights_path')
    reset_optimizer = bool(train_cfg.get('reset_optimizer', False))
    reset_scheduler = bool(train_cfg.get('reset_scheduler', reset_optimizer))

    resume_history = {}
    initial_best_metric = None
    start_epoch = 0
    historical_counter = 0

    if retrain:
        if not pretrained_weights_path:
            raise ValueError("training.retrain is true but training.pretrained_weights_path is empty.")
        resume_history, initial_best_metric, checkpoint = load_resume_artifacts(
            pretrained_weights_path,
            val_key='val_loss',
            train_key='train_loss',
        )
        print(f"Retraining enabled from checkpoint: {pretrained_weights_path}")
        if reset_optimizer:
            print(
                "Optimizer reset enabled: model weights/history will be loaded, "
                f"but AdamW state will start fresh with learning_rate={float(train_cfg['learning_rate']):.6g}."
            )
        if reset_scheduler:
            print("Scheduler reset enabled: scheduler state will start fresh from the current config.")
        if initial_best_metric is not None:
            print(f"Baseline best metric from previous training: {initial_best_metric:.6f}")
        else:
            print("Baseline best metric unavailable from previous training history.")

        partial_model_load = _load_model_state_compatibly(prior, checkpoint['model_state'])
        if partial_model_load:
            reset_optimizer = True
            reset_scheduler = True
            print(
                "Checkpoint architecture differed from the current model; "
                "optimizer and scheduler state will be reset."
            )

        if 'epoch' in checkpoint:
            start_epoch = int(checkpoint['epoch'])
            print(f"Restoring from checkpoint epoch index {start_epoch}.")
        else:
            start_epoch = len(resume_history.get('val_loss', []))
            print(f"Checkpoint did not contain an epoch key. Inferred start_epoch {start_epoch} from history.")

        val_losses_history = resume_history.get('val_loss', [])
        if initial_best_metric is not None and val_losses_history:
            for loss in reversed(val_losses_history):
                if loss > initial_best_metric:
                    historical_counter += 1
                else:
                    break

    epochs = int(train_cfg['epochs'])

    # Initialize EarlyStopping with initial best score
    initial_best_score = None
    if retrain and initial_best_metric is not None:
        initial_best_score = -initial_best_metric  # Note: EarlyStopping score represents -loss

    # Only use early stopping if we have a validation set and a valid initial best score (from previous training or current checkpoint)
    early_stopping = None
    if val_loader is not None:
        early_stopping = EarlyStopping(
            patience=int(train_cfg.get('early_stopping_patience', 10)),
            verbose=True,
            best_score=initial_best_score,
            counter=historical_counter
        )

    adam_beta2 = float(train_cfg.get('adam_beta2',  0.95))
    optimizer = optim.AdamW(
        prior.parameters(),
        lr=float(train_cfg['learning_rate']),
        weight_decay=float(train_cfg.get('weight_decay', 0.01)),
        betas=(0.9, adam_beta2)  # jukebox paper uses default beta1 (0.9) but modified beta2
    )

    if retrain:
        if reset_optimizer:
            print(f"Using fresh optimizer state with learning_rate={float(train_cfg['learning_rate']):.6g}.")
        elif 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            loaded_lrs = [group.get('lr') for group in optimizer.param_groups]
            print(f"Loaded optimizer state from checkpoint. Optimizer LR(s): {loaded_lrs}")
        else:
            print("Checkpoint has no optimizer state; using fresh optimizer state.")

    grad_accum_steps = int(train_cfg.get('gradient_accumulation_steps', 1))
    if grad_accum_steps < 1:
        raise ValueError(f"gradient_accumulation_steps must be >= 1, got {grad_accum_steps}")
    effective_batch_size = int(train_cfg['batch_size']) * grad_accum_steps
    print(
        f"Gradient accumulation: {grad_accum_steps} step(s) "
        f"(micro_batch={int(train_cfg['batch_size'])}, effective_batch={effective_batch_size})"
    )

    scheduler_name = str(train_cfg.get('scheduler', 'onecycle')).strip().lower()
    optimizer_steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum_steps))
    total_steps = optimizer_steps_per_epoch * epochs
    scheduler = None
    if scheduler_name in ('none', 'off', 'disabled'):
        print("Scheduler: disabled")
    elif scheduler_name == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(train_cfg['learning_rate']),
            total_steps=total_steps,
            pct_start=float(train_cfg.get('scheduler_pct_start', 0.05)),
            anneal_strategy=str(train_cfg.get('scheduler_anneal_strategy', 'cos')),
        )
        print(
            f"Scheduler: onecycle "
            f"(optimizer_steps_per_epoch={optimizer_steps_per_epoch}, total_steps={total_steps})"
        )
    else:
        raise ValueError(
            f"Unsupported scheduler '{scheduler_name}'. "
            "Expected one of: onecycle, none"
        )

    if scheduler is not None and retrain:
        if reset_scheduler:
            print("Using fresh scheduler state.")
        elif 'scheduler_state' in checkpoint and checkpoint['scheduler_state'] is not None:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state'])
                print("Loaded scheduler state from checkpoint.")
            except Exception as e:
                print(f"Warning: Failed to load scheduler state ({e}).")
        else:
            print("Checkpoint has no scheduler state; using fresh scheduler state.")

    # Use mixed precision training if on CUDA for potential speedup and reduced memory usage
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    model_name = transformer_cfg['name']
    save_root = train_cfg['save_dir']
    run_dir = os.path.join(
        save_root,
        f"{model_name}_{selected_level}_transformer_prior",
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
    )
    os.makedirs(run_dir, exist_ok=True)

    config_to_save = dict(config)
    config_to_save['vqvae'] = dict(config.get('vqvae', {}))

    bottom_vqvae_dataset_cfg = {}
    existing_dataset_cfg = dict(config.get('dataset', {}))
    merged_dataset_cfg = dict(bottom_vqvae_dataset_cfg)
    merged_dataset_cfg.update(existing_dataset_cfg)
    config_to_save['dataset'] = merged_dataset_cfg

    config_to_save['model'] = dict(config.get('model', {}))
    config_to_save['model']['selected_level'] = selected_level
    config_to_save['model']['inferred_seq_lens'] = dict(seq_lens)
    config_to_save['model']['inferred_grids'] = {
        'top': [int(top_rows), int(top_cols)],
        'middle': [int(middle_rows), int(middle_cols)],
        'bottom': [int(bottom_rows), int(bottom_cols)],
    }
    config_to_save['model']['use_bos_token'] = bool(prior_cfg.get('use_bos_token', False))
    config_to_save['model']['use_timing_conditioning'] = bool(prior_cfg.get('use_timing_conditioning', True))
    config_to_save['model']['use_2d_conditioner'] = bool(prior_cfg.get('use_2d_conditioner', True))
    config_to_save['model']['timing_window_seconds'] = float(
        prior_cfg.get('timing_window_seconds', timing_window_seconds)
    )
    config_to_save['model']['max_time_steps'] = int(max_time_steps)
    if upsample_stride is not None:
        config_to_save['model']['inferred_upsample_stride'] = _serialize_stride(upsample_stride)
    if second_upsample_stride is not None:
        config_to_save['model']['inferred_second_upsample_stride'] = _serialize_stride(second_upsample_stride)
    if cond_seq_len > 0:
        config_to_save['model']['inferred_cond_seq_len'] = int(cond_seq_len)
        config_to_save['model']['inferred_cond_time_cols'] = int(cond_time_cols)
        config_to_save['model']['inferred_cond_freq_bins'] = int(cond_freq_bins)
    if second_cond_seq_len > 0:
        config_to_save['model']['inferred_second_cond_seq_len'] = int(second_cond_seq_len)
        config_to_save['model']['inferred_second_cond_time_cols'] = int(second_cond_time_cols)
        config_to_save['model']['inferred_second_cond_freq_bins'] = int(second_cond_freq_bins)
    config_to_save['dataset'] = dict(config_to_save.get('dataset', {}))
    config_to_save['dataset']['level_target_time_frames'] = dict(level_target_time_frames)

    config_to_save['training'][LEVEL_TO_PRIOR_CFG[selected_level]]['retrain'] = retrain
    config_to_save['training'][LEVEL_TO_PRIOR_CFG[selected_level]]['pretrained_weights_path'] = pretrained_weights_path
    config_to_save['training'][LEVEL_TO_PRIOR_CFG[selected_level]]['reset_optimizer'] = reset_optimizer
    config_to_save['training'][LEVEL_TO_PRIOR_CFG[selected_level]]['reset_scheduler'] = reset_scheduler

    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config_to_save, f)

    train_losses = resume_history.get('train_loss', [])
    val_losses = resume_history.get('val_loss', [])
    best_val_loss = initial_best_metric if initial_best_metric is not None else float('inf')
    best_epoch = start_epoch if start_epoch > 0 else 0

    if early_stopping is not None:
        if (start_epoch >= epochs) or (early_stopping.counter >= early_stopping.patience):
            print(f"Checkpoint epoch ({start_epoch}) is already >= configured epochs ({epochs}) or patience ({early_stopping.patience}) has been exhausted (counter: {early_stopping.counter}). Nothing to train.")
            return

    for epoch in range(start_epoch, epochs):
        prior.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train:{selected_level}]')
        for batch_idx, batch in enumerate(pbar):
            # dataset now returns (target, cond, second_cond, timing) for all levels.
            # 'cond' is the temporally-aligned upper-level slice (None for top).
            # 'second_cond' is the aligned top slice for bottom (None otherwise).
            target_indices = batch[0].to(device)
            cond_indices = batch[1].to(device) if batch[1] is not None else None
            second_cond_indices = batch[2].to(device) if batch[2] is not None else None
            timing = batch[3].to(device)

            # Flatten 2D grids into 1D sequences for the model
            target_seq = target_indices.view(target_indices.shape[0], -1)
            cond_seq = cond_indices.view(cond_indices.shape[0], -1) if cond_indices is not None else None
            second_cond_seq = second_cond_indices.view(second_cond_indices.shape[0], -1) if second_cond_indices is not None else None

            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                    loss = prior.loss(target_seq, upper_indices=cond_seq, second_upper_indices=second_cond_seq, timing=timing)
                loss_to_backward = loss / grad_accum_steps
                scaler.scale(loss_to_backward).backward()
            else:
                loss = prior.loss(target_seq, upper_indices=cond_seq, second_upper_indices=second_cond_seq, timing=timing)
                loss_to_backward = loss / grad_accum_steps
                loss_to_backward.backward()

            should_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(train_loader))
            if should_step:
                if use_amp:
                    scaler.unscale_(optimizer) # unscale gradients before clipping
                    torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_train_loss = running_loss / max(len(train_loader), 1)
        train_losses.append(epoch_train_loss)

        epoch_val_loss = None
        if val_loader is not None:
            prior.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val:{selected_level}]'):
                    target_indices = batch[0].to(device)
                    cond_indices = batch[1].to(device) if batch[1] is not None else None
                    second_cond_indices = batch[2].to(device) if batch[2] is not None else None
                    timing = batch[3].to(device)
                    # Flatten 2D grids (Time, Freq) into 1D sequences (Time*Freq = 512)
                    target_seq = target_indices.view(target_indices.shape[0], -1)
                    cond_seq = cond_indices.view(cond_indices.shape[0], -1) if cond_indices is not None else None
                    second_cond_seq = second_cond_indices.view(second_cond_indices.shape[0], -1) if second_cond_indices is not None else None

                    if use_amp:
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                            loss = prior.loss(target_seq, upper_indices=cond_seq, second_upper_indices=second_cond_seq, timing=timing)
                    else:
                        loss = prior.loss(target_seq, upper_indices=cond_seq, second_upper_indices=second_cond_seq, timing=timing)
                    val_running_loss += loss.item()

            epoch_val_loss = val_running_loss / max(len(val_loader), 1)
            val_losses.append(epoch_val_loss)

        with open(os.path.join(run_dir, 'loss_history.json'), 'w', encoding='utf-8') as f:
            json.dump({'train_loss': train_losses, 'val_loss': val_losses}, f)

        print(
            f"Epoch {epoch + 1}: Train Loss={epoch_train_loss:.4f}"
            + (f" Val Loss={epoch_val_loss:.4f}" if epoch_val_loss is not None else "")
        )

        # Save model state and training artifacts for this epoch (including optimizer and scheduler state for potential resumption)
        model_state = {
            'model_state': prior.state_dict(),
            'config': config_to_save,
            'epoch': epoch + 1,
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'history': {'train_loss': train_losses, 'val_loss': val_losses},
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
        }

        # Save models
        ## not saving every epoch's full checkpoint to avoid excessive storage use, but can be enabled if desired by uncommenting the following lines:
        #torch.save(
        #    model_state,
        #    os.path.join(run_dir, f'model_epoch_{epoch + 1}.pth')
        #)

        ## always save latest model state for potential resumption
        torch.save(
            model_state,
            os.path.join(run_dir, f'latest_model.pth')
        )
        ## Save best model based on validation loss
        if epoch_val_loss is not None:
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = len(val_losses)
                torch.save(
                    {
                        'model_state': prior.state_dict(),
                        'config': config_to_save,
                        'epoch': epoch + 1,
                        'train_loss': epoch_train_loss,
                        'val_loss': epoch_val_loss,
                        'history': {'train_loss': train_losses, 'val_loss': val_losses},
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
                    },
                    os.path.join(run_dir, 'best_model.pth')
                )
                print('Saved best model.')

        if early_stopping is not None:
            early_stopping(epoch_val_loss)
            if early_stopping.early_stop:
                print(f'Early stopping at epoch {epoch + 1}.')
                break

        plot_losses(train_losses, val_losses, run_dir, best_epoch, best_val_loss, level_name=selected_level)

    plot_losses(train_losses, val_losses, run_dir, best_epoch, best_val_loss, level_name=selected_level)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Jukebox Transformer priors (top/middle/bottom).')
    parser.add_argument('--config', type=str, default='./config/config_transformer_prior.yaml')
    parser.add_argument('--level', type=str, default=None, help='Override model.selected_level in config')
    args = parser.parse_args()

    train_transformer_prior(
        args.config,
        level_override=args.level,
    )
