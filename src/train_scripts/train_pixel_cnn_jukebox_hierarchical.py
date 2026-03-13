import argparse
import os
import sys
from datetime import datetime
from typing import Optional, Tuple
import glob

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from callbacks import EarlyStopping
from datasets.jukebox_hierarchical_quantized_dataset import JukeboxHierarchicalQuantizedDataset
from modeling.torch.jukebox_vq_vae import JukeboxVQVAE
from modeling.torch.pixel_cnn_jukebox_levels import JukeboxLevelPixelCNN
from utils import load_config, load_maestro


LEVEL_TO_INT = {'top': 1, 'middle': 2, 'bottom': 3}
LEVEL_TO_PRIOR_CFG = {'top': 'top_prior', 'middle': 'middle_prior', 'bottom': 'bottom_prior'}
COND_LEVEL = {'top': None, 'middle': 'top', 'bottom': 'middle'}
LATEST_ALIASES = {'latest', 'newest', 'auto', 'most_recent'}


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

    candidates = ['module.model.', 'model.module.', 'model.', 'module.']
    for prefix in candidates:
        if all(k.startswith(prefix) for k in keys):
            return {k[len(prefix):]: v for k, v in state_dict.items()}

    model_prefixed = [k for k in keys if k.startswith('model.')]
    if len(model_prefixed) > 0 and len(model_prefixed) >= int(0.8 * len(keys)):
        normalized = {}
        for k, v in state_dict.items():
            normalized[k[len('model.'):]] = v if k.startswith('model.') else v
        return normalized

    return state_dict


def _normalize_state_dict_keys_for_pixelcnn(state_dict: dict) -> dict:
    keys = list(state_dict.keys())
    if not keys:
        return state_dict
    if all(k.startswith('module.') for k in keys):
        return {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict


def _get_prior_cfg(config: dict, name: str) -> dict:
    priors = config.get('priors')
    if priors and name in priors:
        return priors[name]
    return config[name]


def _parse_level(level: str) -> str:
    level = str(level).strip().lower()
    if level in ('mid', 'middle'):
        return 'middle'
    if level not in LEVEL_TO_INT:
        raise ValueError("selected_level must be one of: top, middle, bottom")
    return level


def _extract_num_embeddings_from_state_dict(state_dict: dict) -> Tuple[int, Optional[int]]:
    target_num_embeddings = int(state_dict['pixelcnn_prior.embedding.weight'].shape[0])
    cond_key = 'cond_embedding.weight'
    cond_num_embeddings = int(state_dict[cond_key].shape[0]) if cond_key in state_dict else None
    return target_num_embeddings, cond_num_embeddings


def _get_conditioning_model_path(config: dict, cond_level: str) -> Optional[str]:
    conditioning_cfg = config.get('conditioning_priors', {})
    key = f'{cond_level}_model_dir'
    model_dir = conditioning_cfg.get(key)
    if _is_latest_alias(model_dir):
        return _resolve_latest_prior_dir(config, cond_level)
    return model_dir


def _is_latest_alias(value: Optional[str]) -> bool:
    return isinstance(value, str) and value.strip().lower() in LATEST_ALIASES


def _find_latest_run_dir(parent_dir: str) -> Optional[str]:
    if not os.path.isdir(parent_dir):
        return None

    candidates = []
    for name in os.listdir(parent_dir):
        run_dir = os.path.join(parent_dir, name)
        if not os.path.isdir(run_dir):
            continue
        config_file = os.path.join(run_dir, 'config.yaml')
        if not os.path.isfile(config_file):
            continue
        candidates.append(run_dir)

    if not candidates:
        return None

    # Timestamped run dirs sort correctly lexicographically; fallback by mtime if needed.
    candidates.sort()
    latest = candidates[-1]
    candidates_by_mtime = sorted(candidates, key=lambda p: os.path.getmtime(p))
    if os.path.getmtime(candidates_by_mtime[-1]) > os.path.getmtime(latest):
        latest = candidates_by_mtime[-1]
    return latest


def _resolve_latest_prior_dir(config: dict, level_name: str) -> str:
    model_name = str(config.get('model', {}).get('name', '')).strip()
    save_root = str(config.get('training', {}).get('save_dir', './models/')).strip()

    primary_parent = os.path.join(save_root, f'{model_name}_{level_name}_prior') if model_name else None
    if primary_parent:
        latest = _find_latest_run_dir(primary_parent)
        if latest:
            return latest

    # Fallback: scan save_root for matching prior parent dirs.
    pattern = os.path.join(save_root, f'*_{level_name}_prior')
    parent_dirs = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    latest_candidates = []
    for parent in parent_dirs:
        run_dir = _find_latest_run_dir(parent)
        if run_dir:
            latest_candidates.append(run_dir)

    if latest_candidates:
        latest_candidates.sort(key=lambda p: os.path.getmtime(p))
        return latest_candidates[-1]

    raise FileNotFoundError(
        f"Could not resolve latest prior for level={level_name}. "
        f"Expected runs under '{save_root}' matching '*_{level_name}_prior/<timestamp>/'"
    )


def load_jukebox_model(model_dir_or_file: str, level_name: str, device: torch.device, weights_file: str = 'best_model.pth') -> JukeboxVQVAE:
    model_file, config_path = _resolve_model_file(model_dir_or_file, weights_file)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found at {config_path}')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f'Weights file not found at {model_file}')

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

    print(f'Loading Jukebox {level_name} model from {model_file}')
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    if 'model_state' not in checkpoint:
        raise KeyError(f"Checkpoint at {model_file} does not contain 'model_state'.")

    state_dict = _normalize_state_dict_keys_for_jukebox(checkpoint['model_state'])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        raise RuntimeError(
            f'Error loading Jukebox {level_name} model from {model_file}. '
            f"Missing keys: {missing[:10]}{' ...' if len(missing) > 10 else ''}"
        )
    if unexpected:
        raise RuntimeError(
            f'Error loading Jukebox {level_name} model from {model_file}. '
            f"Unexpected keys: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}"
        )

    model.eval()
    return model


def load_single_level_prior(model_dir_or_file: str, device: torch.device, weights_file: str = 'best_model.pth') -> JukeboxLevelPixelCNN:
    model_file, config_path = _resolve_model_file(model_dir_or_file, weights_file)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found at {config_path}')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f'Weights file not found at {model_file}')

    config = load_config(config_path)
    model_cfg = config.get('model', {})
    selected_level = _parse_level(model_cfg.get('selected_level', 'top'))
    prior_cfg = _get_prior_cfg(config, LEVEL_TO_PRIOR_CFG[selected_level])

    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    if 'model_state' not in checkpoint:
        raise KeyError(f"Checkpoint at {model_file} does not contain 'model_state'.")

    state_dict = _normalize_state_dict_keys_for_pixelcnn(checkpoint['model_state'])
    num_embeddings, cond_num_embeddings = _extract_num_embeddings_from_state_dict(state_dict)

    prior = JukeboxLevelPixelCNN(
        level=LEVEL_TO_INT[selected_level],
        hidden_channels=int(prior_cfg['hidden_channels']),
        num_layers=int(prior_cfg['num_layers']),
        conv_filter_size=int(prior_cfg['conv_filter_size']),
        num_embeddings=num_embeddings,
        cond_num_embeddings=cond_num_embeddings,
        two_level_conditioning_mode=model_cfg.get('two_level_conditioning_mode', 'deconv'),
    ).to(device)

    prior.load_state_dict(state_dict)
    prior.eval()
    return prior


def plot_losses(train_losses, val_losses, save_dir, best_epoch=None, best_val_loss=None):
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')

    if best_epoch is not None and best_val_loss is not None:
        plt.scatter(best_epoch, best_val_loss, c='red', marker='*', s=120, label=f'Best (Loss: {best_val_loss:.4f})', zorder=5)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Jukebox Single-Level PixelCNN Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()


def train_jukebox_hierarchical_pixelcnn(config_path: str, level_override: Optional[str] = None, conditioning_mode_override: Optional[str] = None):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')

    dataset_cfg = config['dataset']
    vqvae_cfg = config['vqvae']
    train_cfg = config['training']
    model_cfg = config.setdefault('model', {})

    selected_level = _parse_level(level_override or model_cfg.get('selected_level', 'top'))
    model_cfg['selected_level'] = selected_level

    conditioning_mode = str(conditioning_mode_override or train_cfg.get('conditioning_mode', 'real')).strip().lower()
    if conditioning_mode not in ('real', 'generated'):
        raise ValueError("training.conditioning_mode must be 'real' or 'generated'")

    top_model = load_jukebox_model(vqvae_cfg['top_model_dir'], 'top', device, vqvae_cfg.get('weights_file', 'best_model.pth'))
    middle_model = load_jukebox_model(vqvae_cfg['middle_model_dir'], 'middle', device, vqvae_cfg.get('weights_file', 'best_model.pth'))
    bottom_model = load_jukebox_model(vqvae_cfg['bottom_model_dir'], 'bottom', device, vqvae_cfg.get('weights_file', 'best_model.pth'))

    x_all, _ = load_maestro(dataset_cfg['processed_path'], dataset_cfg.get('target_time_frames', 256))
    print(f'Loaded data shape: {x_all.shape}')

    quantization_batch_size = train_cfg.get('quantization_batch_size', 32)
    dataset = JukeboxHierarchicalQuantizedDataset(
        x_train=x_all,
        top_model=top_model,
        middle_model=middle_model,
        bottom_model=bottom_model,
        device=device,
        batch_size=quantization_batch_size,
    )

    val_split = train_cfg.get('validation_split', 0.1)
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False)

    prior_cfg = _get_prior_cfg(config, LEVEL_TO_PRIOR_CFG[selected_level])
    num_embeddings_map = {
        'top': int(top_model.vq.num_embeddings),
        'middle': int(middle_model.vq.num_embeddings),
        'bottom': int(bottom_model.vq.num_embeddings),
    }
    cond_level = COND_LEVEL[selected_level]
    cond_num_embeddings = num_embeddings_map[cond_level] if cond_level is not None else None

    # Free VQ-VAE models from GPU once indices are cached.
    del top_model, middle_model, bottom_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    pixelcnn = JukeboxLevelPixelCNN(
        level=LEVEL_TO_INT[selected_level],
        hidden_channels=int(prior_cfg['hidden_channels']),
        num_layers=int(prior_cfg['num_layers']),
        conv_filter_size=int(prior_cfg['conv_filter_size']),
        num_embeddings=num_embeddings_map[selected_level],
        cond_num_embeddings=cond_num_embeddings,
        two_level_conditioning_mode=model_cfg.get('two_level_conditioning_mode', 'deconv'),
    ).to(device)

    conditioning_prior = None
    if cond_level is not None and conditioning_mode == 'generated':
        cond_model_dir = _get_conditioning_model_path(config, cond_level)
        cond_weights_file = config.get('conditioning_priors', {}).get('weights_file', 'best_model.pth')
        if not cond_model_dir:
            raise ValueError(
                f"conditioning_mode=generated for level={selected_level} requires conditioning_priors.{cond_level}_model_dir"
            )
        print(f'Loading conditioning prior ({cond_level}) from {cond_model_dir}')
        conditioning_prior = load_single_level_prior(cond_model_dir, device, cond_weights_file)

    optimizer = optim.Adam(pixelcnn.parameters(), lr=train_cfg['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    model_name = model_cfg['name']
    save_root = train_cfg['save_dir']
    run_dir = os.path.join(save_root, f'{model_name}_{selected_level}_prior', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(run_dir, exist_ok=True)

    config_to_save = dict(config)
    config_to_save['model'] = dict(config.get('model', {}))
    config_to_save['model']['selected_level'] = selected_level
    config_to_save['training'] = dict(config.get('training', {}))
    config_to_save['training']['conditioning_mode'] = conditioning_mode

    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config_to_save, f)

    epochs = int(train_cfg['epochs'])
    early_stopping = EarlyStopping(patience=int(train_cfg.get('early_stopping_patience', 10)), verbose=True)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        pixelcnn.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train:{selected_level}]')
        for batch_indices in pbar:
            top_indices = batch_indices[0].to(device)
            mid_indices = batch_indices[1].to(device)
            bot_indices = batch_indices[2].to(device)

            if selected_level == 'top':
                target_indices = top_indices
                cond_indices = None
            elif selected_level == 'middle':
                target_indices = mid_indices
                cond_indices = top_indices
            else:
                target_indices = bot_indices
                cond_indices = mid_indices

            if conditioning_prior is not None and cond_indices is not None:
                cond_h, cond_w = cond_indices.shape[-2], cond_indices.shape[-1]
                prior_cond = None
                if selected_level == 'bottom':
                    # Middle prior requires top indices as conditioning.
                    prior_cond = top_indices
                with torch.no_grad():
                    cond_indices = conditioning_prior.generate(
                        shape=(target_indices.shape[0], 1, cond_h, cond_w),
                        cond=prior_cond,
                        temperature=1.0,
                        top_k=None,
                    ).squeeze(1)

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = pixelcnn(target_indices, cond=cond_indices)
                    loss = criterion(logits.squeeze(2), target_indices)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = pixelcnn(target_indices, cond=cond_indices)
                loss = criterion(logits.squeeze(2), target_indices)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_train_loss = running_loss / max(len(train_loader), 1)
        train_losses.append(epoch_train_loss)

        pixelcnn.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch_indices in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val:{selected_level}]'):
                top_indices = batch_indices[0].to(device)
                mid_indices = batch_indices[1].to(device)
                bot_indices = batch_indices[2].to(device)

                if selected_level == 'top':
                    target_indices = top_indices
                    cond_indices = None
                elif selected_level == 'middle':
                    target_indices = mid_indices
                    cond_indices = top_indices
                else:
                    target_indices = bot_indices
                    cond_indices = mid_indices

                if conditioning_prior is not None and cond_indices is not None:
                    cond_h, cond_w = cond_indices.shape[-2], cond_indices.shape[-1]
                    prior_cond = None
                    if selected_level == 'bottom':
                        prior_cond = top_indices
                    cond_indices = conditioning_prior.generate(
                        shape=(target_indices.shape[0], 1, cond_h, cond_w),
                        cond=prior_cond,
                        temperature=1.0,
                        top_k=None,
                    ).squeeze(1)

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = pixelcnn(target_indices, cond=cond_indices)
                        loss = criterion(logits.squeeze(2), target_indices)
                else:
                    logits = pixelcnn(target_indices, cond=cond_indices)
                    loss = criterion(logits.squeeze(2), target_indices)

                val_running_loss += loss.item()

        epoch_val_loss = val_running_loss / max(len(val_loader), 1)
        val_losses.append(epoch_val_loss)

        print(f'Epoch {epoch + 1}: Train Loss={epoch_train_loss:.4f}, Val Loss={epoch_val_loss:.4f}')

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            torch.save(
                {
                    'model_state': pixelcnn.state_dict(),
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
                    'model_state': pixelcnn.state_dict(),
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
    parser = argparse.ArgumentParser(description='Train one Jukebox PixelCNN prior level at a time')
    parser.add_argument('--config', type=str, default='./config/config_pixelcnn_jukebox_hierarchical.yaml')
    parser.add_argument('--level', type=str, default=None, choices=['top', 'middle', 'bottom'])
    parser.add_argument('--conditioning_mode', type=str, default=None, choices=['real', 'generated'])
    args = parser.parse_args()

    train_jukebox_hierarchical_pixelcnn(args.config, level_override=args.level, conditioning_mode_override=args.conditioning_mode)
