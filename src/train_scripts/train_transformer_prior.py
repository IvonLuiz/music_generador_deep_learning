import os
import sys
import yaml
import argparse
from datetime import datetime
from typing import Optional

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
from utils import load_maestro, load_config
from callbacks import EarlyStopping
from train_scripts.jukebox_utils import load_jukebox_model, _parse_level


LEVEL_TO_PRIOR_CFG = {'top': 'top_prior', 'middle': 'middle_prior', 'bottom': 'bottom_prior'}
COND_LEVEL = {'top': None, 'middle': 'top', 'bottom': 'middle'}


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
    train_cfg = config['training']

    selected_level = _parse_level(level_override or transformer_cfg.get('selected_level', 'top'))
    transformer_cfg['selected_level'] = selected_level

    effective_weights_file = weights_file or vqvae_cfg.get('weights_file', 'best_model.pth')
    effective_top_model_dir = top_model_dir or vqvae_cfg['top_model_dir']
    effective_middle_model_dir = middle_model_dir or vqvae_cfg['middle_model_dir']
    effective_bottom_model_dir = bottom_model_dir or vqvae_cfg['bottom_model_dir']

    top_model = load_jukebox_model(effective_top_model_dir, 'top', device, effective_weights_file)
    middle_model = load_jukebox_model(effective_middle_model_dir, 'middle', device, effective_weights_file)
    bottom_model = load_jukebox_model(effective_bottom_model_dir, 'bottom', device, effective_weights_file)

    x_all, _ = load_maestro(dataset_cfg['processed_path'], dataset_cfg.get('target_time_frames', 256))
    print(f'Loaded data shape: {x_all.shape}')

    quant_batch_size = train_cfg.get('quantization_batch_size', 32)
    dataset = JukeboxHierarchicalQuantizedDataset(
        x_train=x_all,
        top_model=top_model,
        middle_model=middle_model,
        bottom_model=bottom_model,
        device=device,
        batch_size=quant_batch_size,
    )

    num_embeddings_map = {
        'top': int(top_model.vq.num_embeddings),
        'middle': int(middle_model.vq.num_embeddings),
        'bottom': int(bottom_model.vq.num_embeddings),
    }

    del top_model, middle_model, bottom_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    top_h, top_w = dataset.top_indices.shape[-2], dataset.top_indices.shape[-1]
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

    val_split = train_cfg.get('validation_split', 0.1)
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False)

    prior_cfg = _get_prior_cfg(config, LEVEL_TO_PRIOR_CFG[selected_level])
    cond_level = COND_LEVEL[selected_level]

    is_upsampler = cond_level is not None
    cond_num_embeddings = num_embeddings_map[cond_level] if is_upsampler else None
    upsample_stride = None
    if is_upsampler:
        upsample_stride = _compute_stride(seq_lens[selected_level], seq_lens[cond_level], selected_level)

    prior = TransformerPriorConditioned(
        num_embeddings=num_embeddings_map[selected_level],
        model_dim=int(prior_cfg['model_dim']),
        num_heads=int(prior_cfg['num_heads']),
        num_layers=int(prior_cfg['num_layers']),
        dim_feedforward=int(prior_cfg['dim_feedforward']),
        max_seq_len=seq_lens[selected_level],
        block_len=int(prior_cfg.get('block_len', 16)),
        is_upsampler=is_upsampler,
        cond_num_embeddings=cond_num_embeddings,
        upsample_stride=upsample_stride,
        dropout=float(prior_cfg.get('dropout', 0.1)),
    ).to(device)

    optimizer = optim.AdamW(prior.parameters(), lr=float(train_cfg['learning_rate']), weight_decay=float(train_cfg.get('weight_decay', 0.01)))

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
    config_to_save['model'] = dict(config.get('model', {}))
    config_to_save['model']['selected_level'] = selected_level
    config_to_save['model']['inferred_seq_lens'] = dict(seq_lens)
    config_to_save['model']['inferred_grids'] = {
        'top': [int(top_h), int(top_w)],
        'middle': [int(middle_h), int(middle_w)],
        'bottom': [int(bottom_h), int(bottom_w)],
    }
    if upsample_stride is not None:
        config_to_save['model']['inferred_upsample_stride'] = int(upsample_stride)

    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config_to_save, f)

    epochs = int(train_cfg['epochs'])
    early_stopping = EarlyStopping(patience=int(train_cfg.get('early_stopping_patience', 10)), verbose=True)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        prior.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train:{selected_level}]')
        for batch in pbar:
            top_indices = batch[0].to(device)
            mid_indices = batch[1].to(device)
            bot_indices = batch[2].to(device)

            if selected_level == 'top':
                target_indices = top_indices
                cond_indices = None
            elif selected_level == 'middle':
                target_indices = mid_indices
                cond_indices = top_indices
            else:
                target_indices = bot_indices
                cond_indices = mid_indices

            target_seq = target_indices.view(target_indices.shape[0], -1)
            cond_seq = cond_indices.view(cond_indices.shape[0], -1) if cond_indices is not None else None
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    loss = prior.loss(target_seq, upper_indices=cond_seq)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = prior.loss(target_seq, upper_indices=cond_seq)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
                optimizer.step()

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

                if selected_level == 'top':
                    target_indices = top_indices
                    cond_indices = None
                elif selected_level == 'middle':
                    target_indices = mid_indices
                    cond_indices = top_indices
                else:
                    target_indices = bot_indices
                    cond_indices = mid_indices

                target_seq = target_indices.view(target_indices.shape[0], -1)
                cond_seq = cond_indices.view(cond_indices.shape[0], -1) if cond_indices is not None else None
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        loss = prior.loss(target_seq, upper_indices=cond_seq)
                else:
                    loss = prior.loss(target_seq, upper_indices=cond_seq)
                val_running_loss += loss.item()

        epoch_val_loss = val_running_loss / max(len(val_loader), 1)
        val_losses.append(epoch_val_loss)

        print(f'Epoch {epoch + 1}: Train Loss={epoch_train_loss:.4f}, Val Loss={epoch_val_loss:.4f}')

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            torch.save(
                {
                    'model_state': prior.state_dict(),
                    'config': config_to_save,
                    'epoch': epoch + 1,
                    'train_loss': epoch_train_loss,
                    'val_loss': epoch_val_loss,
                },
                os.path.join(run_dir, 'best_model.pth')
            )
            print('Saved best model.')

        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    'model_state': prior.state_dict(),
                    'epoch': epoch + 1,
                    'train_loss': epoch_train_loss,
                    'val_loss': epoch_val_loss,
                },
                os.path.join(run_dir, f'model_epoch_{epoch + 1}.pth')
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
