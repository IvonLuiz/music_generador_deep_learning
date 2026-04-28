import os
import sys
import yaml
import argparse
import json
import numpy as np
from datetime import datetime
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.jukebox_hierarchical_quantized_dataset import JukeboxHierarchicalQuantizedDataset
from modeling.torch.transformer_prior_conditioned import TransformerPriorConditioned
from utils import set_global_seed, list_npy_files, load_config
from callbacks import EarlyStopping
from train_scripts.jukebox_utils import load_jukebox_model, parse_level
from train_scripts.resume_utils import load_resume_artifacts

LEVEL_TO_PRIOR_CFG = {'top': 'top_prior', 'middle': 'middle_prior', 'bottom': 'bottom_prior'}
COND_LEVEL = {'top': None, 'middle': 'top', 'bottom': 'middle'}
SECOND_COND_LEVEL = {'top': None, 'middle': None, 'bottom': 'top'}


def plot_losses(train_losses, val_losses, save_dir, best_epoch=None, best_val_loss=None, level_name: str = 'top'):
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')

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


def _resolve_vqvae_config_path(model_dir_or_file: Optional[str]) -> Optional[str]:
    if not model_dir_or_file:
        return None
    if os.path.isdir(model_dir_or_file):
        cfg = os.path.join(model_dir_or_file, 'config.yaml')
        return cfg if os.path.exists(cfg) else None
    if os.path.isfile(model_dir_or_file):
        name = os.path.basename(model_dir_or_file).lower()
        if name in ('config.yaml', 'config.yml'):
            return model_dir_or_file
        cfg = os.path.join(os.path.dirname(model_dir_or_file), 'config.yaml')
        return cfg if os.path.exists(cfg) else None
    return None

def train_transformer_prior(
    config_path: str,
    level_override: Optional[str] = None,
    top_model_dir: Optional[str] = None,
    middle_model_dir: Optional[str] = None,
    bottom_model_dir: Optional[str] = None,
    weights_file: Optional[str] = None,
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

    effective_weights_file = weights_file or vqvae_cfg.get('weights_file', 'best_model.pth')
    effective_top_model_dir = top_model_dir or vqvae_cfg['top_model_dir']
    effective_middle_model_dir = middle_model_dir or vqvae_cfg['middle_model_dir']
    effective_bottom_model_dir = bottom_model_dir or vqvae_cfg['bottom_model_dir']

    top_model = load_jukebox_model(effective_top_model_dir, 'top', device, effective_weights_file)
    middle_model = load_jukebox_model(effective_middle_model_dir, 'middle', device, effective_weights_file)
    bottom_model = load_jukebox_model(effective_bottom_model_dir, 'bottom', device, effective_weights_file)

    target_time_frames = int(dataset_cfg.get('target_time_frames', 2048))
    level_target_time_frames = dataset_cfg.get('level_target_time_frames') or {}
    sample_rate = int(dataset_cfg.get('sample_rate', 22050))
    hop_length = int(dataset_cfg.get('hop_length', 256))
    segment_overlap = float(dataset_cfg.get('segment_overlap', 0.0))

    file_paths = list_npy_files(dataset_cfg['processed_path'])
    if len(file_paths) == 0:
        raise ValueError(f"No .npy files found under dataset path: {dataset_cfg['processed_path']}")
    print(f'Found {len(file_paths)} spectrogram files for quantization.')
    print(f'Level target time frames: {level_target_time_frames}')

    quant_batch_size = train_cfg.get('quantization_batch_size', 8)
    dataset = JukeboxHierarchicalQuantizedDataset(
        file_paths=file_paths,
        top_model=top_model,
        middle_model=middle_model,
        bottom_model=bottom_model,
        device=device,
        batch_size=quant_batch_size,
        target_time_frames=target_time_frames,
        level_target_time_frames=level_target_time_frames,
        sample_rate=sample_rate,
        hop_length=hop_length,
        segment_overlap=segment_overlap,
    )

    num_embeddings_map = {
        'top': int(top_model.vq.num_embeddings),
        'middle': int(middle_model.vq.num_embeddings),
        'bottom': int(bottom_model.vq.num_embeddings),
    }

    del top_model, middle_model, bottom_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Grid dims: stored as (N, Time, Freq) after the transpose in the dataset.
    top_h, top_w = dataset.top_indices.shape[-2], dataset.top_indices.shape[-1]       # Time, Freq
    middle_h, middle_w = dataset.middle_indices.shape[-2], dataset.middle_indices.shape[-1]
    bottom_h, bottom_w = dataset.bottom_indices.shape[-2], dataset.bottom_indices.shape[-1]
    seq_lens = {
        'top': int(top_h * top_w),
        'middle': int(middle_h * middle_w),
        'bottom': int(bottom_h * bottom_w),
    }
    print(
        f"Top indices shape: {tuple(dataset.top_indices.shape)} -> seq_len={seq_lens['top']}"
    )
    print(
        f"Middle indices shape: {tuple(dataset.middle_indices.shape)} -> seq_len={seq_lens['middle']}"
    )
    print(
        f"Bottom indices shape: {tuple(dataset.bottom_indices.shape)} -> seq_len={seq_lens['bottom']}"
    )

    # --- Temporal alignment for conditioning ---
    # Each upsampler trains on a SHORT window of the target level, conditioned on the
    # corresponding SLICE of the upper level that covers exactly the same audio.
    # The slice is always the first cond_time_cols time-columns of the upper grid.
    #
    #   Middle conditioned on Top slice:
    #     cond_time_cols = top_h * (mid_tf / top_tf)    e.g. 64 * (512/2048) = 16
    #     cond_seq_len   = cond_time_cols * top_w        e.g. 16 * 8 = 128 → stride 512/128 = 4
    #
    #   Bottom conditioned on Middle slice (primary):
    #     cond_time_cols = mid_h * (bot_tf / mid_tf)    e.g. 32 * (128/512) = 8
    #     cond_seq_len   = cond_time_cols * mid_w        e.g. 8 * 16 = 128 → stride 512/128 = 4
    #
    #   Bottom conditioned on Top slice (secondary):
    #     second_cond_time_cols = top_h * (bot_tf / top_tf)  e.g. 64 * (128/2048) = 4
    #     second_cond_seq_len   = second_cond_time_cols * top_w  e.g. 4 * 8 = 32 → stride 512/32 = 16
    #
    # Note: because ConvTranspose1d(kernel_size=stride) maps input token i to output
    # tokens [i*stride, (i+1)*stride), the WaveNetConditioner's existing
    # cond_emb[:, :seq_len, :] slice in forward() produces the SAME result as if we had
    # passed only cond_seq_len tokens — but we slice EXPLICITLY here during training so
    # the model never receives out-of-alignment conditioning.

    _top_tf = int(level_target_time_frames.get('top', target_time_frames))
    _mid_tf = int(level_target_time_frames.get('middle', target_time_frames))
    _bot_tf = int(level_target_time_frames.get('bottom', target_time_frames))

    cond_time_cols: int = 0
    second_cond_time_cols: int = 0
    cond_seq_len: int = 0
    second_cond_seq_len: int = 0
    cond_grid_h: int = 0
    cond_grid_w: int = 0

    if selected_level == 'middle':
        cond_time_cols = round(top_h * _mid_tf / _top_tf)
        cond_grid_h, cond_grid_w = top_h, top_w
        cond_seq_len = cond_time_cols * top_w
    elif selected_level == 'bottom':
        cond_time_cols = round(middle_h * _bot_tf / _mid_tf)
        cond_grid_h, cond_grid_w = middle_h, middle_w
        cond_seq_len = cond_time_cols * middle_w
        if condition_on_top:
            second_cond_time_cols = round(top_h * _bot_tf / _top_tf)
            second_cond_seq_len = second_cond_time_cols * top_w

    if cond_seq_len > 0:
        print(f"Conditioning slice for {selected_level}: {cond_time_cols} time-cols × {cond_grid_w} freq = {cond_seq_len} tokens")
    if second_cond_seq_len > 0:
        print(f"Second conditioning slice (Top→Bottom): {second_cond_time_cols} time-cols × {top_w} freq = {second_cond_seq_len} tokens")

    # Split into train/val
    split_rng = np.random.default_rng(seed)
    split_rng.shuffle(dataset.file_paths)
    val_split = train_cfg.get('validation_split', 0.1)
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False)

    prior_cfg = _get_prior_cfg(config, LEVEL_TO_PRIOR_CFG[selected_level])
    cond_level = COND_LEVEL[selected_level]
    second_cond_level = SECOND_COND_LEVEL[selected_level]
    condition_on_top = bool(prior_cfg.get('condition_on_top', False)) and selected_level == 'bottom'

    is_upsampler = cond_level is not None
    upsample_stride = None
    cond_num_embeddings = None
    second_upsample_stride = None
    second_cond_num_embeddings = None
    if is_upsampler:
        cond_num_embeddings = num_embeddings_map[cond_level]
        # Stride is computed from the SLICED conditioning length, not the full upper seq_len.
        # With equal seq_lens (all 512) the old full-seq computation would yield stride=1.
        upsample_stride = _compute_stride(seq_lens[selected_level], cond_seq_len, selected_level)
    if condition_on_top and second_cond_level is not None:
        second_cond_num_embeddings = num_embeddings_map[second_cond_level]
        second_upsample_stride = _compute_stride(seq_lens[selected_level], second_cond_seq_len, selected_level)

    max_time_steps = int(prior_cfg.get('max_time_steps', 500))

    prior = TransformerPriorConditioned(
        num_embeddings=num_embeddings_map[selected_level],
        model_dim=int(prior_cfg['model_dim']),
        num_heads=int(prior_cfg['num_heads']),
        num_layers=int(prior_cfg['num_layers']),
        dim_feedforward=int(prior_cfg['dim_feedforward']),
        max_seq_len=seq_lens[selected_level],
        block_len=int(prior_cfg.get('block_len', 16)),
        max_time_steps=max_time_steps,
        is_upsampler=is_upsampler,
        cond_num_embeddings=cond_num_embeddings,
        upsample_stride=upsample_stride,
        second_cond_num_embeddings=second_cond_num_embeddings,
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
    ).to(device)

    retrain = bool(train_cfg.get('retrain', False))
    pretrained_weights_path = train_cfg.get('pretrained_weights_path')

    resume_history = {}
    initial_best_metric = None
    start_epoch = 0
    historical_counter = 0

    if retrain:
        if not pretrained_weights_path:
            raise ValueError("training.retrain is true but training.pretrained_weights_path is empty.")
        resume_history, initial_best_metric, checkpoint = load_resume_artifacts(pretrained_weights_path, val_key='val_loss', train_key='train_loss')
        print(f"Retraining enabled from checkpoint: {pretrained_weights_path}")
        if initial_best_metric is not None:
            print(f"Baseline best metric from previous training: {initial_best_metric:.6f}")
        else:
            print("Baseline best metric unavailable from previous training history.")

        prior.load_state_dict(checkpoint['model_state'])

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
        
    early_stopping = EarlyStopping(
        patience=int(train_cfg.get('early_stopping_patience', 10)), 
        verbose=True,
        best_score=initial_best_score,
        counter=historical_counter
    )

    optimizer = optim.AdamW(prior.parameters(), lr=float(train_cfg['learning_rate']), weight_decay=float(train_cfg.get('weight_decay', 0.01)))

    if retrain and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    grad_accum_steps = int(train_cfg.get('gradient_accumulation_steps', 1))
    if grad_accum_steps < 1:
        raise ValueError(f"gradient_accumulation_steps must be >= 1, got {grad_accum_steps}")
    effective_batch_size = int(train_cfg['batch_size']) * grad_accum_steps
    print(
        f"Gradient accumulation: {grad_accum_steps} step(s) "
        f"(micro_batch={int(train_cfg['batch_size'])}, effective_batch={effective_batch_size})"
    )

    # Warmup for the first ~5% of total training steps
    optimizer_steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum_steps))
    total_steps = optimizer_steps_per_epoch * epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(train_cfg['learning_rate']),
        total_steps=total_steps,
        pct_start=0.05, # Peaks at 5% of training, then gently decays
        anneal_strategy='cos'
    )
    
    if retrain and 'scheduler_state' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
        except Exception as e:
            print(f"Warning: Failed to load scheduler state ({e}).")

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
    config_to_save['vqvae'] = dict(config['vqvae'])
    config_to_save['vqvae']['top_model_dir'] = effective_top_model_dir
    config_to_save['vqvae']['middle_model_dir'] = effective_middle_model_dir
    config_to_save['vqvae']['bottom_model_dir'] = effective_bottom_model_dir
    config_to_save['vqvae']['weights_file'] = effective_weights_file

    bottom_vqvae_cfg_path = _resolve_vqvae_config_path(effective_bottom_model_dir)
    bottom_vqvae_dataset_cfg = {}
    if bottom_vqvae_cfg_path is not None:
        try:
            bottom_vqvae_cfg = load_config(bottom_vqvae_cfg_path)
            if isinstance(bottom_vqvae_cfg, dict):
                bottom_vqvae_dataset_cfg = dict(bottom_vqvae_cfg.get('dataset', {}))
        except (FileNotFoundError, OSError, yaml.YAMLError, ValueError) as exc:
            print(
                f"Warning: failed to load bottom VQ-VAE config from {bottom_vqvae_cfg_path}: {exc}",
                file=sys.stderr,
            )

    existing_dataset_cfg = dict(config.get('dataset', {}))
    merged_dataset_cfg = dict(bottom_vqvae_dataset_cfg)
    merged_dataset_cfg.update(existing_dataset_cfg)
    config_to_save['dataset'] = merged_dataset_cfg

    config_to_save['model'] = dict(config.get('model', {}))
    config_to_save['model']['selected_level'] = selected_level
    config_to_save['model']['inferred_seq_lens'] = dict(seq_lens)
    config_to_save['model']['inferred_grids'] = {
        'top': [int(top_h), int(top_w)],
        'middle': [int(middle_h), int(middle_w)],
        'bottom': [int(bottom_h), int(bottom_w)],
    }
    config_to_save['model']['use_bos_token'] = bool(prior_cfg.get('use_bos_token', False))
    config_to_save['model']['max_time_steps'] = int(max_time_steps)
    if upsample_stride is not None:
        config_to_save['model']['inferred_upsample_stride'] = int(upsample_stride)
    if second_upsample_stride is not None:
        config_to_save['model']['inferred_second_upsample_stride'] = int(second_upsample_stride)
    # Persist conditioning slice info for diagnostics and potential future use.
    if cond_seq_len > 0:
        config_to_save['model']['inferred_cond_seq_len'] = int(cond_seq_len)
        config_to_save['model']['inferred_cond_time_cols'] = int(cond_time_cols)
    if second_cond_seq_len > 0:
        config_to_save['model']['inferred_second_cond_seq_len'] = int(second_cond_seq_len)
        config_to_save['model']['inferred_second_cond_time_cols'] = int(second_cond_time_cols)
    config_to_save['dataset'] = dict(config_to_save.get('dataset', {}))
    config_to_save['dataset']['level_target_time_frames'] = dict(level_target_time_frames)

    config_to_save['training'][LEVEL_TO_PRIOR_CFG[selected_level]]['retrain'] = retrain
    config_to_save['training'][LEVEL_TO_PRIOR_CFG[selected_level]]['pretrained_weights_path'] = pretrained_weights_path

    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config_to_save, f)

    train_losses = resume_history.get('train_loss', [])
    val_losses = resume_history.get('val_loss', [])
    best_val_loss = initial_best_metric if initial_best_metric is not None else float('inf')
    best_epoch = start_epoch if start_epoch > 0 else 0

    if start_epoch >= epochs or early_stopping.counter >= early_stopping.patience:
        print(f"Checkpoint epoch ({start_epoch}) is already >= configured epochs ({epochs}) or patience ({early_stopping.patience}) has been exhausted (counter: {early_stopping.counter}). Nothing to train.")
        return

    for epoch in range(start_epoch, epochs):
        prior.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train:{selected_level}]')
        for batch_idx, batch in enumerate(pbar):
            top_indices = batch[0].to(device)
            mid_indices = batch[1].to(device)
            bot_indices = batch[2].to(device)
            timing = batch[3].to(device)  # (B, 3): [start_time_s, total_duration_s, fraction_elapsed]

            if selected_level == 'top':
                target_indices = top_indices
                cond_indices = None
                second_cond_indices = None
            elif selected_level == 'middle':
                target_indices = mid_indices
                cond_indices = top_indices
                second_cond_indices = None
            else:
                target_indices = bot_indices
                cond_indices = mid_indices
                second_cond_indices = top_indices if condition_on_top else None

            target_seq = target_indices.view(target_indices.shape[0], -1)

            # Slice upper-level conditioning to the temporally-aligned window.
            # Stored grids are (B, Time, Freq); we keep the first cond_time_cols time-steps.
            if cond_indices is not None:
                B = cond_indices.shape[0]
                cond_seq = cond_indices.view(B, cond_grid_h, cond_grid_w)[:, :cond_time_cols, :].reshape(B, -1)
            else:
                cond_seq = None

            if second_cond_indices is not None:
                B = second_cond_indices.shape[0]
                second_cond_seq = second_cond_indices.view(B, top_h, top_w)[:, :second_cond_time_cols, :].reshape(B, -1)
            else:
                second_cond_seq = None

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

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_train_loss = running_loss / max(len(train_loader), 1)
        train_losses.append(epoch_train_loss)

        prior.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val:{selected_level}]'):
                top_indices = batch[0].to(device)
                mid_indices = batch[1].to(device)
                bot_indices = batch[2].to(device)
                timing = batch[3].to(device)

                if selected_level == 'top':
                    target_indices = top_indices
                    cond_indices = None
                    second_cond_indices = None
                elif selected_level == 'middle':
                    target_indices = mid_indices
                    cond_indices = top_indices
                    second_cond_indices = None
                else:
                    target_indices = bot_indices
                    cond_indices = mid_indices
                    second_cond_indices = top_indices if condition_on_top else None

                target_seq = target_indices.view(target_indices.shape[0], -1)

                if cond_indices is not None:
                    B = cond_indices.shape[0]
                    cond_seq = cond_indices.view(B, cond_grid_h, cond_grid_w)[:, :cond_time_cols, :].reshape(B, -1)
                else:
                    cond_seq = None

                if second_cond_indices is not None:
                    B = second_cond_indices.shape[0]
                    second_cond_seq = second_cond_indices.view(B, top_h, top_w)[:, :second_cond_time_cols, :].reshape(B, -1)
                else:
                    second_cond_seq = None

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

        print(f'Epoch {epoch + 1}: Train Loss={epoch_train_loss:.4f}, Val Loss={epoch_val_loss:.4f}')

        # Save models
        ## Save best model based on validation loss
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
                },
                os.path.join(run_dir, 'best_model.pth')
            )
            print('Saved best model.')

        ## Periodic checkpointing every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    'model_state': prior.state_dict(),
                    'epoch': epoch + 1,
                    'train_loss': epoch_train_loss,
                    'val_loss': epoch_val_loss,
                    'history': {'train_loss': train_losses, 'val_loss': val_losses},
                },
                os.path.join(run_dir, f'model_epoch_{epoch + 1}.pth')
            )

        ## Always save the latest checkpoint
        torch.save(
            {
                'model_state': prior.state_dict(),
                'epoch': epoch + 1,
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'history': {'train_loss': train_losses, 'val_loss': val_losses},
            },
            os.path.join(run_dir, f'latest_model.pth')
        )

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
    parser.add_argument('--top_model_dir', type=str, default=None, help='Override vqvae.top_model_dir from config')
    parser.add_argument('--middle_model_dir', type=str, default=None, help='Override vqvae.middle_model_dir from config')
    parser.add_argument('--bottom_model_dir', type=str, default=None, help='Override vqvae.bottom_model_dir from config')
    parser.add_argument('--weights_file', type=str, default=None, help='Override vqvae.weights_file from config')
    args = parser.parse_args()

    train_transformer_prior(
        args.config,
        level_override=args.level,
        top_model_dir=args.top_model_dir,
        middle_model_dir=args.middle_model_dir,
        bottom_model_dir=args.bottom_model_dir,
        weights_file=args.weights_file,
    )
