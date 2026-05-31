import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Sampler, Subset
from tqdm import tqdm

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from callbacks import EarlyStopping
from datasets.quantized_dataset import PixelCNNQuantizedDataset, QuantizedDataset
from modeling.torch.pixel_cnn import ConditionalGatedPixelCNN
from utils import load_config, load_maestro, load_vqvae_model


def _select_global_train_parity(mode: str, epoch: int, seed: int) -> str:
    if mode == 'all':
        return 'all'
    if mode == 'alternate':
        return 'even' if epoch % 2 == 0 else 'odd'
    if mode == 'random_per_song':
        return 'random_per_song'
    raise ValueError(f"Unsupported train window parity mode: {mode}")


def _indices_for_train_epoch(dataset, mode: str, epoch: int, seed: int):
    parity = _select_global_train_parity(mode, epoch, seed)
    if parity == 'random_per_song' and hasattr(dataset, 'indices_for_random_window_parity_per_song'):
        return dataset.indices_for_random_window_parity_per_song(seed + epoch), parity
    if hasattr(dataset, 'indices_for_window_parity'):
        return dataset.indices_for_window_parity(parity), parity
    return list(range(len(dataset))), 'all'


class WindowParityEpochSampler(Sampler):
    """Shuffle a fixed overlap-parity subset, changing that subset each epoch."""

    def __init__(self, dataset, mode: str, seed: int):
        self.dataset = dataset
        self.mode = str(mode or 'alternate').strip().lower()
        self.seed = int(seed)
        self.epoch = 0
        self.current_parity = 'all'
        self.indices = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        indices, parity = _indices_for_train_epoch(self.dataset, self.mode, self.epoch, self.seed)
        rng = np.random.default_rng(self.seed + self.epoch)
        self.indices = list(indices)
        rng.shuffle(self.indices)
        self.current_parity = parity

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def plot_pixelcnn_losses(train_losses, val_losses, save_dir):
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PixelCNN Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()


def save_loss_history(train_losses, val_losses, save_dir):
    history = {
        'train_loss': list(train_losses),
        'val_loss': list(val_losses),
    }
    with open(os.path.join(save_dir, 'loss_history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    return history


def _loader_kwargs(train_cfg):
    num_workers = int(train_cfg.get('num_workers', 0))
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': bool(train_cfg.get('pin_memory', torch.cuda.is_available())),
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = bool(train_cfg.get('persist_workers', True))
        prefetch_factor = train_cfg.get('prefetch_factor', 4)
        if prefetch_factor is not None:
            kwargs['prefetch_factor'] = int(prefetch_factor)
    return kwargs


def _resolve_num_embeddings(pixelcnn_config, train_dataset, vqvae_model_path, device):
    model_cfg = pixelcnn_config.get('model', {})
    dataset_cfg = pixelcnn_config.get('dataset', {})
    for cfg in (model_cfg, dataset_cfg):
        for key in ('K', 'num_embeddings'):
            if cfg.get(key) is not None:
                return int(cfg[key])

    if getattr(train_dataset, 'num_embeddings', None) is not None:
        return int(train_dataset.num_embeddings)

    if vqvae_model_path:
        print(f"Loading VQ-VAE only to infer codebook size: {vqvae_model_path}")
        vqvae = load_vqvae_model(vqvae_model_path, device)
        K = int(vqvae.vq.num_embeddings)
        del vqvae
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return K

    raise ValueError(
        "Could not determine PixelCNN num_embeddings. Set model.K/model.num_embeddings, "
        "dataset.num_embeddings, or pass a VQ-VAE model path."
    )


def _build_precomputed_loaders(pixelcnn_config, train_cfg):
    dataset_cfg = pixelcnn_config['dataset']
    quantized_path = dataset_cfg.get('quantized_path')
    if not quantized_path:
        raise ValueError("dataset.quantized_path is required when dataset.input_mode='quantized'.")

    manifest_file = dataset_cfg.get('manifest_file', 'pixelcnn_quantized_manifest.jsonl')
    preload = bool(dataset_cfg.get('preload', False))
    train_splits = dataset_cfg.get('train_splits', ['train'])
    validation_splits = dataset_cfg.get(
        'validation_splits',
        [dataset_cfg.get('validation_split_name', 'validation')],
    )
    train_window_parity_mode = dataset_cfg.get('train_window_parity_mode', 'alternate')
    validation_window_parity = dataset_cfg.get('validation_window_parity', 'even')

    train_dataset = PixelCNNQuantizedDataset(
        quantized_path=quantized_path,
        split=train_splits,
        manifest_file=manifest_file,
        preload=preload,
    )
    val_dataset = PixelCNNQuantizedDataset(
        quantized_path=quantized_path,
        split=validation_splits,
        manifest_file=manifest_file,
        preload=preload,
    )

    batch_size = int(train_cfg['batch_size'])
    kwargs = _loader_kwargs(train_cfg)
    train_sampler = WindowParityEpochSampler(
        train_dataset,
        mode=train_window_parity_mode,
        seed=int(train_cfg.get('seed', 42)),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, **kwargs)

    if hasattr(train_dataset, 'window_parity_counts'):
        counts = train_dataset.window_parity_counts()
        print(
            f"Training window parity mode: {train_window_parity_mode} "
            f"(all={counts['all']}, even={counts['even']}, odd={counts['odd']})"
        )

    if hasattr(val_dataset, 'indices_for_window_parity'):
        val_indices = val_dataset.indices_for_window_parity(validation_window_parity)
    else:
        val_indices = list(range(len(val_dataset)))
    val_dataset_for_loader = Subset(val_dataset, val_indices)
    print(
        f"Validation window parity: {validation_window_parity} "
        f"(active={len(val_dataset_for_loader)}, full={len(val_dataset)})"
    )
    val_loader = DataLoader(val_dataset_for_loader, batch_size=batch_size, shuffle=False, **kwargs)
    return train_dataset, val_dataset_for_loader, train_loader, val_loader, train_sampler


def _build_legacy_loaders(pixelcnn_config, vqvae_model_path, device):
    dataset_cfg = pixelcnn_config['dataset']
    train_cfg = pixelcnn_config['training']
    if not vqvae_model_path:
        raise ValueError("A VQ-VAE model path is required for legacy spectrogram quantization.")

    print(f"Loading VQ-VAE model from {vqvae_model_path}")
    vqvae = load_vqvae_model(vqvae_model_path, device)

    spectrograms_path = dataset_cfg['processed_path']
    target_time_frames = int(dataset_cfg.get('target_time_frames', 256))
    spectrograms_data, _ = load_maestro(spectrograms_path, target_time_frames)
    if len(spectrograms_data) == 0:
        raise ValueError(
            f"No spectrograms were loaded from {spectrograms_path}. "
            "Use dataset.input_mode='quantized' with dataset.quantized_path for the precomputed workflow."
        )

    validation_split = float(train_cfg.get('validation_split', 0.1))
    num_samples = len(spectrograms_data)
    num_val = int(num_samples * validation_split)
    num_train = num_samples - num_val
    indices = np.random.permutation(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    x_train = spectrograms_data[train_indices]
    x_val = spectrograms_data[val_indices]
    print(f"Data split: {len(x_train)} training, {len(x_val)} validation samples.")

    quantization_batch_size = int(train_cfg.get('quantization_batch_size', 32))
    print(f"Creating legacy Training Dataset (quantization_batch_size={quantization_batch_size})...")
    train_dataset = QuantizedDataset(x_train, vqvae, device, batch_size=quantization_batch_size)
    print(f"Creating legacy Validation Dataset (quantization_batch_size={quantization_batch_size})...")
    val_dataset = QuantizedDataset(x_val, vqvae, device, batch_size=quantization_batch_size)

    batch_size = int(train_cfg['batch_size'])
    kwargs = _loader_kwargs(train_cfg)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    return train_dataset, val_dataset, train_loader, val_loader, None


def train_pixel_cnn(pixelcnn_config_path: str, vqvae_model_path: str = None):
    pixelcnn_config = load_config(pixelcnn_config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    dataset_cfg = pixelcnn_config['dataset']
    train_cfg = pixelcnn_config['training']
    model_cfg = pixelcnn_config['model']

    input_mode = str(
        dataset_cfg.get('input_mode', 'quantized' if dataset_cfg.get('quantized_path') else 'spectrogram')
    ).strip().lower()
    if input_mode not in ('quantized', 'spectrogram'):
        raise ValueError("dataset.input_mode must be either 'quantized' or 'spectrogram'.")

    if input_mode == 'quantized':
        train_dataset, val_dataset, train_loader, val_loader, train_sampler = _build_precomputed_loaders(pixelcnn_config, train_cfg)
    else:
        train_dataset, val_dataset, train_loader, val_loader, train_sampler = _build_legacy_loaders(pixelcnn_config, vqvae_model_path, device)

    K = _resolve_num_embeddings(pixelcnn_config, train_dataset, vqvae_model_path, device)
    print(f"Using K={K} codebook entries")
    print(f"Data split windows: {len(train_dataset)} training, {len(val_dataset)} validation")

    pixel_cnn = ConditionalGatedPixelCNN(
        in_channels=1,
        hidden_channels=int(model_cfg['hidden_channels']),
        num_layers=int(model_cfg['num_layers']),
        kernel_size=int(model_cfg['kernel_size']),
        num_classes=K,
        num_embeddings=K,
    ).to(device)

    optimizer = optim.Adam(pixel_cnn.parameters(), lr=float(train_cfg['learning_rate']))
    criterion = nn.CrossEntropyLoss()
    use_amp = bool(train_cfg.get('amp', True)) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path_dir = os.path.join(train_cfg['save_dir'], f"{model_cfg['name']}", current_time)
    os.makedirs(save_path_dir, exist_ok=True)

    config_to_save = dict(pixelcnn_config)
    config_to_save['dataset'] = dict(dataset_cfg)
    config_to_save['dataset']['input_mode'] = input_mode
    config_to_save['model'] = dict(model_cfg)
    config_to_save['model']['K'] = K
    config_to_save['training'] = dict(train_cfg)
    config_to_save['vqvae_model_path'] = vqvae_model_path

    with open(os.path.join(save_path_dir, "config.yaml"), 'w') as f:
        yaml.dump(config_to_save, f)

    epochs = int(train_cfg['epochs'])
    save_every = int(train_cfg.get('save_every', 10))
    early_stopping = EarlyStopping(
        patience=int(train_cfg.get('early_stopping_patience', 10)),
        verbose=True,
    )

    print(f"Starting PixelCNN training for {epochs} epochs...")
    print(f"Saving artifacts to: {save_path_dir}")

    best_pixelcnn_model_path = os.path.join(save_path_dir, "best_pixelcnn_model.pth")
    best_model_path = os.path.join(save_path_dir, "best_model.pth")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch - 1)

        pixel_cnn.train()
        total_train_loss = 0.0

        parity_suffix = f":{train_sampler.current_parity}" if train_sampler is not None else ""
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train{parity_suffix}]")
        for batch_indices in pbar:
            batch_indices = batch_indices.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    output = pixel_cnn(batch_indices).squeeze(2)
                    loss = criterion(output, batch_indices)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(pixel_cnn.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = pixel_cnn(batch_indices).squeeze(2)
                loss = criterion(output, batch_indices)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pixel_cnn.parameters(), 1.0)
                optimizer.step()

            total_train_loss += loss.item()
            pbar.set_postfix(loss=total_train_loss / max(1, pbar.n + 1))

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        train_losses.append(avg_train_loss)

        pixel_cnn.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")
            for batch_indices in val_bar:
                batch_indices = batch_indices.to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        output = pixel_cnn(batch_indices).squeeze(2)
                        loss = criterion(output, batch_indices)
                else:
                    output = pixel_cnn(batch_indices).squeeze(2)
                    loss = criterion(output, batch_indices)
                total_val_loss += loss.item()
                val_bar.set_postfix(val_loss=total_val_loss / max(1, val_bar.n + 1))

        avg_val_loss = total_val_loss / max(len(val_loader), 1)
        val_losses.append(avg_val_loss)

        history = save_loss_history(train_losses, val_losses, save_path_dir)
        plot_pixelcnn_losses(train_losses, val_losses, save_path_dir)

        print(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        checkpoint_payload = {
            'model_state': pixel_cnn.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'config': config_to_save,
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'history': history,
        }

        if save_every > 0 and (epoch % save_every == 0 or epoch == epochs):
            save_file = os.path.join(save_path_dir, f"pixelcnn_epoch_{epoch}.pth")
            torch.save(checkpoint_payload, save_file)
            print(f"Saved checkpoint to {save_file}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint_payload, best_pixelcnn_model_path)
            torch.save(checkpoint_payload, best_model_path)
            print(f"Saved best model on validation data to {best_pixelcnn_model_path} on epoch {epoch}")

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("Training complete.")


def _find_latest_vqvae_model(config_path: str = "./config/config_vqvae.yaml"):
    vqvae_global_config = load_config(config_path)
    save_dir = vqvae_global_config['training']['save_dir']
    model_path = None
    if os.path.exists(save_dir):
        valid_runs = []
        for name in os.listdir(save_dir):
            run_dir = os.path.join(save_dir, name)
            if not os.path.isdir(run_dir):
                continue
            candidates = ["best_model.pth", "model.pth"]
            if any(os.path.exists(os.path.join(run_dir, candidate)) for candidate in candidates):
                valid_runs.append(run_dir)

        if valid_runs:
            model_path = max(valid_runs, key=os.path.getmtime)
            print(f"Found latest VQ-VAE model run: {model_path}")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PixelCNN on single VQ-VAE code indices.")
    parser.add_argument("--config", type=str, default="./config/config_pixelcnn.yaml")
    parser.add_argument("--vqvae_model", type=str, default=None, help="VQ-VAE run dir/checkpoint, only needed for legacy mode or K fallback.")
    parser.add_argument("--vqvae_config", type=str, default="./config/config_vqvae.yaml")
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    model_path = args.vqvae_model
    if model_path is None:
        model_path = _find_latest_vqvae_model(args.vqvae_config)

    train_pixel_cnn(args.config, model_path)
