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

    train_file_paths, val_file_paths = split_train_val_paths(
        all_file_paths=all_file_paths,
        dataset_cfg=dataset_cfg,
        validation_split=train_cfg.get('validation_split', 0.1),
        seed=seed,
    )
    train_file_paths=train_file_paths[:64]  # --- TEMPORARY LIMIT FOR TESTING ---
    val_file_paths=val_file_paths[:]       # --- TEMPORARY LIMIT FOR

    # Load precomputed quantized dataset
    print(f"--- Loading Precomputed Quantized Dataset for {selected_level} ---")
    train_dataset = JukeboxQuantizedDataset(
        quantized_path=quantized_data_path,
        file_paths=train_file_paths,
        target_time_frames=target_time_frames,
    )
    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True)
    val_loader = None
    if val_file_paths:
        print(f"--- Loading Precomputed Validation Dataset for {selected_level} ---")
        val_dataset = JukeboxQuantizedDataset(
            quantized_path=quantized_data_path,
            file_paths=val_file_paths,
            target_time_frames=target_time_frames,
        )
        val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False)

    # Use explicit codebook size (not inferred from data, which may have smaller max token)
    num_embeddings_map = {
        'top': vqvae_codebook_size,
        'middle': vqvae_codebook_size,
        'bottom': vqvae_codebook_size,
    }
    print(f"Using codebook size: {vqvae_codebook_size}")

    sample = train_dataset[0]
    top_indices, middle_indices, bottom_indices, timing, sample_meta = sample
    top_rows, top_cols = top_indices.shape
    middle_rows, middle_cols = (middle_indices.shape if middle_indices.numel() > 0 else (0, 0))
    bottom_rows, bottom_cols = (bottom_indices.shape if bottom_indices.numel() > 0 else (0, 0))

    target_tensor = top_indices if selected_level == 'top' else middle_indices if selected_level == 'middle' else bottom_indices
    target_seq_len = int(target_tensor.numel())
    seq_lens = {'top': target_seq_len, 'middle': target_seq_len, 'bottom': target_seq_len}

    print(f"Top grid shape: {tuple(top_indices.shape)}")
    print(f"Middle grid shape: {tuple(middle_indices.shape)}")
    print(f"Bottom grid shape: {tuple(bottom_indices.shape)}")

    # --- Temporal alignment for conditioning ---
    # Each upsampler trains on a window of the target level, conditioned on the
    # matching slice of the upper level that covers the same audio span.
    prior_cfg = _get_prior_cfg(config, LEVEL_TO_PRIOR_CFG[selected_level])
    cond_level = COND_LEVEL[selected_level]
    second_cond_level = SECOND_COND_LEVEL[selected_level]
    condition_on_top = bool(prior_cfg.get('condition_on_top', False)) and selected_level == 'bottom'

    _top_tf = int(level_target_time_frames.get('top', target_time_frames))
    _mid_tf = int(level_target_time_frames.get('middle', target_time_frames))
    _bot_tf = int(level_target_time_frames.get('bottom', target_time_frames))

    cond_time_cols = 0
    second_cond_time_cols = 0
    cond_seq_len = 0
    second_cond_seq_len = 0
    cond_grid_h = 0
    cond_grid_w = 0

    if selected_level == 'middle':
        cond_time_cols = max(1, round(top_rows * _mid_tf / _top_tf))
        cond_grid_h, cond_grid_w = top_rows, top_cols
        cond_seq_len = cond_time_cols * top_cols
    elif selected_level == 'bottom':
        cond_time_cols = max(1, round(middle_rows * _bot_tf / _mid_tf))
        cond_grid_h, cond_grid_w = middle_rows, middle_cols
        cond_seq_len = cond_time_cols * middle_cols
        if condition_on_top:
            second_cond_time_cols = max(1, round(top_rows * _bot_tf / _top_tf))
            second_cond_seq_len = second_cond_time_cols * top_cols

    if cond_seq_len > 0:
        print(f"Conditioning slice for {selected_level}: {cond_time_cols} time-cols × {cond_grid_w} freq = {cond_seq_len} tokens")
    if second_cond_seq_len > 0:
        print(f"Second conditioning slice (Top→Bottom): {second_cond_time_cols} time-cols × {top_cols} freq = {second_cond_seq_len} tokens")

    is_upsampler = cond_level is not None
    upsample_stride = None
    cond_num_embeddings = None
    second_upsample_stride = None
    second_cond_num_embeddings = None
    if is_upsampler:
        cond_num_embeddings = num_embeddings_map[cond_level]
        upsample_stride = _compute_stride(target_seq_len, cond_seq_len, selected_level)
    if condition_on_top and second_cond_level is not None:
        second_cond_num_embeddings = num_embeddings_map[second_cond_level]
        second_upsample_stride = _compute_stride(target_seq_len, second_cond_seq_len, selected_level)

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
        resume_history, initial_best_metric, checkpoint = load_resume_artifacts(
            pretrained_weights_path,
            val_key='val_loss',
            train_key='train_loss',
        )
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

    # Only use early stopping if we have a validation set and a valid initial best score (from previous training or current checkpoint)
    early_stopping = None
    if val_loader is not None and initial_best_score is not None:
        early_stopping = EarlyStopping(
            patience=int(train_cfg.get('early_stopping_patience', 10)), 
            verbose=True,
            best_score=initial_best_score,
            counter=historical_counter
        )

    adam_beta2 = float(prior_cfg.get('adam_beta2', 0.95))
    optimizer = optim.AdamW(
        prior.parameters(), 
        lr=float(train_cfg['learning_rate']), 
        weight_decay=float(train_cfg.get('weight_decay', 0.01)),
        betas=(0.9, adam_beta2) # jukebox paper uses default beta1 (0.9) but modified beta2
    )

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

    if early_stopping is not None :
        if (start_epoch >= epochs) or (early_stopping.counter >= early_stopping.patience):
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
            meta = batch[4].to(device) if len(batch) > 4 else torch.zeros((top_indices.shape[0], 2), device=device, dtype=torch.long)

            cond_seq = None
            second_cond_seq = None
            if selected_level == 'top':
                target_indices = top_indices
            elif selected_level == 'middle':
                target_indices = mid_indices
                cond_seq = top_indices.view(top_indices.shape[0], -1) # condition on the aligned slice of Top (already handled by Dataset)
            else: # bottom
                target_indices = bot_indices
                cond_seq = mid_indices.view(mid_indices.shape[0], -1)
                second_cond_seq = top_indices.view(top_indices.shape[0], -1)

            target_seq = target_indices.view(target_indices.shape[0], -1)   # flatten the target grid for loss

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

        epoch_val_loss = None
        if val_loader is not None:
            prior.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val:{selected_level}]'):
                    top_indices = batch[0].to(device)
                    mid_indices = batch[1].to(device)
                    bot_indices = batch[2].to(device)
                    timing = batch[3].to(device)
                    meta = batch[4].to(device) if len(batch) > 4 else torch.zeros((top_indices.shape[0], 2), device=device, dtype=torch.long)

                    # Flatten 2D grids (Time, Freq) into 1D sequences (Time*Freq = 512)
                    top_seq = top_indices.view(top_indices.shape[0], -1)
                    mid_seq = mid_indices.view(mid_indices.shape[0], -1)
                    bot_seq = bot_indices.view(bot_indices.shape[0], -1)

                    cond_seq = None
                    second_cond_seq = None
                    if selected_level == 'top':
                        target_seq = top_seq
                    elif selected_level == 'middle':
                        target_seq = mid_seq
                        # Condition on the aligned slice of Top
                        cond_seq = top_seq 
                    else: # bottom
                        target_seq = bot_seq
                        # Primary conditioning (Middle) and Secondary (Top)
                        cond_seq = mid_seq
                        second_cond_seq = top_seq

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
    
        if epoch_val_loss:
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
