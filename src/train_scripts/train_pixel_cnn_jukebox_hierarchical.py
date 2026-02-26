import os
import sys
import yaml
import argparse
from datetime import datetime
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling.torch.jukebox_vq_vae import JukeboxVQVAE
from modeling.torch.pixel_cnn_hierarchical import HierarchicalCondGatedPixelCNN
from datasets.jukebox_hierarchical_quantized_dataset import JukeboxHierarchicalQuantizedDataset
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
    plt.title('Jukebox 3-Level PixelCNN Training and Validation Loss')
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
    """
    Normalize checkpoint keys for JukeboxVQVAE.

    Handles checkpoints saved from wrappers such as:
    - JukeboxHierarchicalAdapter(model=...): keys start with "model."
    - DataParallel wrappers: keys start with "module."
    - Combined wrappers: "module.model." / "model.module."
    """
    keys = list(state_dict.keys())
    if not keys:
        return state_dict

    candidates = [
        "module.model.",
        "model.module.",
        "model.",
        "module.",
    ]

    for prefix in candidates:
        if all(k.startswith(prefix) for k in keys):
            return {k[len(prefix):]: v for k, v in state_dict.items()}

    # Fallback: if most keys start with model., strip only those.
    model_prefixed = [k for k in keys if k.startswith("model.")]
    if len(model_prefixed) > 0 and len(model_prefixed) >= int(0.8 * len(keys)):
        normalized = {}
        for k, v in state_dict.items():
            normalized[k[len("model."):]] = v if k.startswith("model.") else v
        return normalized

    return state_dict


def load_jukebox_model(model_dir_or_file: str, level_name: str, device: torch.device, weights_file: str = 'best_model.pth') -> JukeboxVQVAE:
    model_file, config_path = _resolve_model_file(model_dir_or_file, weights_file)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Weights file not found at {model_file}")

    config = load_config(config_path)
    model_cfg = config['model']

    level_profiles = model_cfg.get('level_profiles', {})
    if level_name not in level_profiles:
        raise ValueError(f"Level '{level_name}' not found in level_profiles of {config_path}")

    level_profile = level_profiles[level_name]
    levels = level_profile['levels']
    num_residual_layers = level_profile.get('num_residual_layers', 4)

    activation_name = str(model_cfg.get('activation', '')).lower()
    activation_layer = nn.Sigmoid() if activation_name == 'sigmoid' else None

    model = JukeboxVQVAE(
        input_channels=model_cfg['input_channels'],
        hidden_dim=model_cfg['hidden_dim'],
        levels=levels,
        num_residual_layers=num_residual_layers,
        num_embeddings=model_cfg.get('num_embeddings', 2048),
        embedding_dim=model_cfg.get('embedding_dim', 64),
        beta=model_cfg.get('beta', 0.25),
        conv_type=model_cfg.get('conv_type', 2),
        activation_layer=activation_layer,
        dilation_growth_rate=model_cfg.get('dilation_growth_rate', 3),
        channel_growth=model_cfg.get('channel_growth', 1),
    ).to(device)

    print(f"Loading Jukebox {level_name} model from {model_file}")
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    if 'model_state' not in checkpoint:
        raise KeyError(f"Checkpoint at {model_file} does not contain 'model_state'.")

    state_dict = _normalize_state_dict_keys_for_jukebox(checkpoint['model_state'])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        raise RuntimeError(
            f"Error loading Jukebox {level_name} model from {model_file}. "
            f"Missing keys: {missing[:10]}{' ...' if len(missing) > 10 else ''}"
        )
    if unexpected:
        raise RuntimeError(
            f"Error loading Jukebox {level_name} model from {model_file}. "
            f"Unexpected keys: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}"
        )

    model.eval()
    return model


def train_jukebox_hierarchical_pixelcnn(config_path: str):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    dataset_cfg = config['dataset']
    vqvae_cfg = config['vqvae']
    train_cfg = config['training']

    top_model = load_jukebox_model(vqvae_cfg['top_model_dir'], 'top', device, vqvae_cfg.get('weights_file', 'best_model.pth'))
    middle_model = load_jukebox_model(vqvae_cfg['middle_model_dir'], 'middle', device, vqvae_cfg.get('weights_file', 'best_model.pth'))
    bottom_model = load_jukebox_model(vqvae_cfg['bottom_model_dir'], 'bottom', device, vqvae_cfg.get('weights_file', 'best_model.pth'))

    x_all, _ = load_maestro(dataset_cfg['processed_path'], dataset_cfg.get('target_time_frames', 256))
    print(f"Loaded data shape: {x_all.shape}")

    quantization_batch_size = train_cfg.get('quantization_batch_size', 32)
    dataset = JukeboxHierarchicalQuantizedDataset(
        x_train=x_all,
        top_model=top_model,
        middle_model=middle_model,
        bottom_model=bottom_model,
        device=device,
        batch_size=quantization_batch_size,
    )

    example_top, example_mid, example_bot = dataset[0]
    input_size = [tuple(example_top.shape), tuple(example_mid.shape), tuple(example_bot.shape)]
    print(f"Inferred latent sizes (top/mid/bottom): {input_size}")

    val_split = train_cfg.get('validation_split', 0.1)
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False)

    top_cfg = config['top_prior']
    mid_cfg = config['middle_prior']
    bot_cfg = config['bottom_prior']

    hidden_units = [top_cfg['hidden_channels'], mid_cfg['hidden_channels'], bot_cfg['hidden_channels']]
    num_layers = [top_cfg['num_layers'], mid_cfg['num_layers'], bot_cfg['num_layers']]
    conv_filter_size = [top_cfg['conv_filter_size'], mid_cfg['conv_filter_size'], bot_cfg['conv_filter_size']]
    dropout = [top_cfg.get('dropout_rate', 0.0), mid_cfg.get('dropout_rate', 0.0), bot_cfg.get('dropout_rate', 0.0)]
    num_embeddings = [top_model.vq.num_embeddings, middle_model.vq.num_embeddings, bottom_model.vq.num_embeddings]

    pixelcnn = HierarchicalCondGatedPixelCNN(
        num_prior_levels=3,
        input_size=input_size,
        hidden_units=hidden_units,
        num_layers=num_layers,
        conv_filter_size=conv_filter_size,
        dropout=dropout,
        num_embeddings=num_embeddings,
        residual_units=[1024, 1024, 1024],
        attention_layers=[0, 0, 0],
        attention_heads=[None, None, None],
        conditioning_stack_residual_blocks=[None, 20, 20],
    ).to(device)

    optimizer = optim.Adam(pixelcnn.parameters(), lr=train_cfg['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    model_name = config['model']['name']
    save_root = train_cfg['save_dir']
    run_dir = os.path.join(save_root, f"{model_name}_jukebox_hierarchical_pixelcnn", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    epochs = train_cfg['epochs']
    early_stopping = EarlyStopping(patience=train_cfg.get('early_stopping_patience', 10), verbose=True)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        pixelcnn.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for batch_indices in pbar:
            top_indices = batch_indices[0].to(device)
            mid_indices = batch_indices[1].to(device)
            bot_indices = batch_indices[2].to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits_top = pixelcnn(top_indices, level='top')
                    loss_top = criterion(logits_top.squeeze(2), top_indices)

                    logits_mid = pixelcnn(mid_indices, cond=top_indices, level='mid')
                    loss_mid = criterion(logits_mid.squeeze(2), mid_indices)

                    logits_bot = pixelcnn(bot_indices, cond=mid_indices, level='bottom')
                    loss_bot = criterion(logits_bot.squeeze(2), bot_indices)

                    loss = loss_top + loss_mid + loss_bot

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits_top = pixelcnn(top_indices, level='top')
                loss_top = criterion(logits_top.squeeze(2), top_indices)

                logits_mid = pixelcnn(mid_indices, cond=top_indices, level='mid')
                loss_mid = criterion(logits_mid.squeeze(2), mid_indices)

                logits_bot = pixelcnn(bot_indices, cond=mid_indices, level='bottom')
                loss_bot = criterion(logits_bot.squeeze(2), bot_indices)

                loss = loss_top + loss_mid + loss_bot
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'top': f"{loss_top.item():.4f}",
                'mid': f"{loss_mid.item():.4f}",
                'bot': f"{loss_bot.item():.4f}",
            })

        epoch_train_loss = running_loss / max(len(train_loader), 1)
        train_losses.append(epoch_train_loss)

        pixelcnn.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch_indices in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                top_indices = batch_indices[0].to(device)
                mid_indices = batch_indices[1].to(device)
                bot_indices = batch_indices[2].to(device)

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        logits_top = pixelcnn(top_indices, level='top')
                        loss_top = criterion(logits_top.squeeze(2), top_indices)

                        logits_mid = pixelcnn(mid_indices, cond=top_indices, level='mid')
                        loss_mid = criterion(logits_mid.squeeze(2), mid_indices)

                        logits_bot = pixelcnn(bot_indices, cond=mid_indices, level='bottom')
                        loss_bot = criterion(logits_bot.squeeze(2), bot_indices)
                else:
                    logits_top = pixelcnn(top_indices, level='top')
                    loss_top = criterion(logits_top.squeeze(2), top_indices)

                    logits_mid = pixelcnn(mid_indices, cond=top_indices, level='mid')
                    loss_mid = criterion(logits_mid.squeeze(2), mid_indices)

                    logits_bot = pixelcnn(bot_indices, cond=mid_indices, level='bottom')
                    loss_bot = criterion(logits_bot.squeeze(2), bot_indices)

                val_running_loss += (loss_top + loss_mid + loss_bot).item()

        epoch_val_loss = val_running_loss / max(len(val_loader), 1)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch + 1}: Train Loss={epoch_train_loss:.4f}, Val Loss={epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            torch.save(
                {
                    'model_state': pixelcnn.state_dict(),
                    'config': config,
                    'epoch': epoch + 1,
                    'train_loss': epoch_train_loss,
                    'val_loss': epoch_val_loss,
                },
                os.path.join(run_dir, 'best_model.pth')
            )
            print("Saved best model.")

        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    'model_state': pixelcnn.state_dict(),
                    'epoch': epoch + 1,
                    'train_loss': epoch_train_loss,
                    'val_loss': epoch_val_loss,
                },
                os.path.join(run_dir, f"model_epoch_{epoch + 1}.pth")
            )

        early_stopping(epoch_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

        plot_losses(train_losses, val_losses, run_dir, best_epoch, best_val_loss)

    plot_losses(train_losses, val_losses, run_dir, best_epoch, best_val_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 3-level Jukebox-style hierarchical PixelCNN priors')
    parser.add_argument('--config', type=str, default='./config/config_pixelcnn_jukebox_hierarchical.yaml')
    args = parser.parse_args()

    train_jukebox_hierarchical_pixelcnn(args.config)
