import os
import sys
import yaml
import argparse
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling.torch.jukebox_vq_vae import JukeboxVQVAE
from modeling.torch.transformer_prior import TransformerPrior
from utils import load_maestro, load_config
from callbacks import EarlyStopping


def plot_losses(train_losses, val_losses, save_dir, best_epoch=None, best_val_loss=None):
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')

    if best_epoch is not None and best_val_loss is not None:
        plt.scatter(best_epoch, best_val_loss, c='red', marker='*', s=120, label=f'Best (Loss: {best_val_loss:.4f})', zorder=5)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Transformer Top Prior Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()


def _resolve_model_file(model_dir_or_file: str, weights_file: str) -> Tuple[str, str]:
    if os.path.isfile(model_dir_or_file):
        model_file = model_dir_or_file
        config_path = os.path.join(os.path.dirname(model_file), 'config.yaml')
    else:
        model_file = os.path.join(model_dir_or_file, weights_file)
        config_path = os.path.join(model_dir_or_file, 'config.yaml')
    return model_file, config_path


def _normalize_state_dict_keys_for_jukebox(state_dict: dict) -> dict:
    keys = list(state_dict.keys())
    if not keys:
        return state_dict

    candidates = [
        'module.model.',
        'model.module.',
        'model.',
        'module.',
    ]

    for prefix in candidates:
        if all(k.startswith(prefix) for k in keys):
            return {k[len(prefix):]: v for k, v in state_dict.items()}

    return state_dict


def load_jukebox_top_model(model_dir_or_file: str, device: torch.device, weights_file: str = 'best_model.pth') -> JukeboxVQVAE:
    model_file, config_path = _resolve_model_file(model_dir_or_file, weights_file)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found at {config_path}')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f'Weights file not found at {model_file}')

    config = load_config(config_path)
    model_cfg = config['model']
    level_profile = model_cfg.get('level_profiles', {}).get('top')
    if level_profile is None:
        raise ValueError(f"Missing model.level_profiles.top in {config_path}")

    activation_name = str(model_cfg.get('activation', '')).lower()
    activation_layer = nn.Sigmoid() if activation_name == 'sigmoid' else None

    model = JukeboxVQVAE(
        input_channels=model_cfg['input_channels'],
        hidden_dim=model_cfg['hidden_dim'],
        levels=level_profile['levels'],
        num_residual_layers=level_profile.get('num_residual_layers', 4),
        num_embeddings=model_cfg.get('num_embeddings', 2048),
        embedding_dim=model_cfg.get('embedding_dim', 64),
        beta=model_cfg.get('beta', 0.25),
        conv_type=model_cfg.get('conv_type', 2),
        activation_layer=activation_layer,
        dilation_growth_rate=model_cfg.get('dilation_growth_rate', 3),
        channel_growth=model_cfg.get('channel_growth', 1),
    ).to(device)

    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    if 'model_state' not in checkpoint:
        raise KeyError(f"Checkpoint at {model_file} does not contain 'model_state'.")

    state_dict = _normalize_state_dict_keys_for_jukebox(checkpoint['model_state'])
    model.load_state_dict(state_dict)
    model.eval()
    return model


def extract_top_sequences(x_all: np.ndarray, top_model: JukeboxVQVAE, device: torch.device, batch_size: int):
    all_indices = []
    with torch.no_grad():
        for i in tqdm(range(0, len(x_all), batch_size), desc='Extracting top-level codes'):
            batch = x_all[i:i + batch_size]
            x_t = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(device)
            z = top_model.encoder(x_t)
            z = top_model.pre_vq_conv(z)
            _, idx, _, _, _ = top_model.vq(z)
            all_indices.append(idx.cpu().long())

    indices_2d = torch.cat(all_indices, dim=0)
    sequences = indices_2d.view(indices_2d.shape[0], -1)
    return indices_2d, sequences


def train_transformer_top_prior(config_path: str, top_model_dir: str = None, weights_file: str = None):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')

    dataset_cfg = config['dataset']
    vqvae_cfg = config['vqvae']
    model_cfg = config['model']
    train_cfg = config['training']

    effective_top_model_dir = top_model_dir or vqvae_cfg['top_model_dir']
    effective_weights_file = weights_file or vqvae_cfg.get('weights_file', 'best_model.pth')

    top_model = load_jukebox_top_model(
        model_dir_or_file=effective_top_model_dir,
        device=device,
        weights_file=effective_weights_file,
    )

    x_all, _ = load_maestro(dataset_cfg['processed_path'], dataset_cfg.get('target_time_frames', 256))
    print(f'Loaded data shape: {x_all.shape}')

    quant_batch_size = train_cfg.get('quantization_batch_size', 32)
    top_indices_2d, top_sequences = extract_top_sequences(x_all, top_model, device, quant_batch_size)
    _, top_h, top_w = top_indices_2d.shape
    seq_len = int(top_h * top_w)
    print(f'Top indices shape: {tuple(top_indices_2d.shape)} -> sequence length {seq_len}')

    val_split = train_cfg.get('validation_split', 0.1)
    dataset = TensorDataset(top_sequences)
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False)

    prior = TransformerPrior(
        num_embeddings=int(top_model.vq.num_embeddings),
        model_dim=int(model_cfg['model_dim']),
        num_heads=int(model_cfg['num_heads']),
        num_layers=int(model_cfg['num_layers']),
        dim_feedforward=int(model_cfg['dim_feedforward']),
        max_seq_len=seq_len,
        dropout=float(model_cfg.get('dropout', 0.1)),
    ).to(device)

    optimizer = optim.AdamW(prior.parameters(), lr=float(train_cfg['learning_rate']), weight_decay=float(train_cfg.get('weight_decay', 0.01)))

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    model_name = model_cfg['name']
    save_root = train_cfg['save_dir']
    run_dir = os.path.join(save_root, f"{model_name}_transformer_top_prior", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(run_dir, exist_ok=True)

    config_to_save = dict(config)
    config_to_save['vqvae'] = dict(config['vqvae'])
    config_to_save['vqvae']['top_model_dir'] = effective_top_model_dir
    config_to_save['vqvae']['weights_file'] = effective_weights_file
    config_to_save['model'] = dict(config['model'])
    config_to_save['model']['inferred_seq_len'] = seq_len
    config_to_save['model']['inferred_top_grid'] = [int(top_h), int(top_w)]

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

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')
        for batch in pbar:
            indices = batch[0].to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    loss = prior.loss(indices)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = prior.loss(indices)
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
            for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]'):
                indices = batch[0].to(device)
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        loss = prior.loss(indices)
                else:
                    loss = prior.loss(indices)
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

        plot_losses(train_losses, val_losses, run_dir, best_epoch, best_val_loss)

    plot_losses(train_losses, val_losses, run_dir, best_epoch, best_val_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train top-level Transformer prior on Jukebox VQ indices')
    parser.add_argument('--config', type=str, default='./config/config_transformer_prior_top.yaml')
    parser.add_argument('--top_model_dir', type=str, default=None, help='Override vqvae.top_model_dir from config')
    parser.add_argument('--weights_file', type=str, default=None, help='Override vqvae.weights_file from config')
    args = parser.parse_args()

    train_transformer_top_prior(args.config, top_model_dir=args.top_model_dir, weights_file=args.weights_file)
