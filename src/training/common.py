from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from callbacks import SampleGenerator


@dataclass
class DataBundle:
    """!
    @brief Container returned by data modules and consumed by TrainingEngine.

    It groups dataloaders with optional metadata needed by adapters. For
    example, VQ-VAE adapters use data_variance for reconstruction scaling, while
    PixelCNN adapters use num_embeddings and input_size from quantized manifests.
    """

    train_loader: DataLoader
    val_loader: Optional[DataLoader] = None
    train_dataset: Any = None
    val_dataset: Any = None
    data_variance: Optional[float] = None
    batch_preprocessor: Optional[torch.nn.Module] = None
    collate_fn: Any = None
    num_embeddings: Any = None
    input_size: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def get_training_cfg(config: dict) -> dict:
    """!
    @brief Return config.training, creating it when absent.
    """
    return config.setdefault('training', {})


def get_callbacks_cfg(config: dict) -> dict:
    """!
    @brief Return callback settings while accepting legacy training keys.
    """
    training_cfg = get_training_cfg(config)
    callbacks_cfg = config.setdefault('callbacks', {})
    if 'early_stopping_patience' not in callbacks_cfg and 'early_stopping_patience' in training_cfg:
        callbacks_cfg['early_stopping_patience'] = training_cfg['early_stopping_patience']
    if 'save_every' not in callbacks_cfg and 'save_every' in training_cfg:
        callbacks_cfg['save_every'] = training_cfg['save_every']
    return callbacks_cfg


def get_resume_cfg(config: dict) -> dict:
    """!
    @brief Return resume settings while accepting legacy training keys.
    """
    training_cfg = get_training_cfg(config)
    resume_cfg = config.setdefault('resume', {})
    if 'enabled' not in resume_cfg and 'retrain' in training_cfg:
        resume_cfg['enabled'] = bool(training_cfg.get('retrain', False))
    if 'checkpoint_path' not in resume_cfg and training_cfg.get('pretrained_weights_path'):
        resume_cfg['checkpoint_path'] = training_cfg.get('pretrained_weights_path')
    if 'reset_optimizer' not in resume_cfg and 'reset_optimizer' in training_cfg:
        resume_cfg['reset_optimizer'] = bool(training_cfg.get('reset_optimizer', False))
    if 'reset_scheduler' not in resume_cfg and 'reset_scheduler' in training_cfg:
        resume_cfg['reset_scheduler'] = bool(training_cfg.get('reset_scheduler', False))
    return resume_cfg


def create_run_dir(save_dir: str, run_subdir: Optional[str] = None) -> str:
    """!
    @brief Create a timestamped artifact directory for one training run.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    parts = [save_dir]
    if run_subdir:
        parts.append(run_subdir)
    parts.append(timestamp)
    run_dir = os.path.join(*parts)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def dataloader_kwargs(training_cfg: dict) -> dict:
    """!
    @brief Build DataLoader keyword arguments shared by all training scripts.
    """
    num_workers = int(training_cfg.get('num_workers', 0))
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': bool(training_cfg.get('pin_memory', torch.cuda.is_available())),
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = bool(training_cfg.get('persist_workers', True))
        prefetch_factor = training_cfg.get('prefetch_factor', 4)
        if prefetch_factor is not None:
            kwargs['prefetch_factor'] = int(prefetch_factor)
    return kwargs


def move_to_device(batch, device: torch.device):
    """!
    @brief Recursively move tensors in a nested batch structure to device.
    """
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {
            key: move_to_device(value, device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
    if isinstance(batch, tuple):
        return tuple(move_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    return batch


def normalize_history(history: Optional[dict]) -> dict:
    """!
    @brief Normalize old and new loss-history key names into one dictionary.
    """
    if not isinstance(history, dict):
        return {}
    normalized = {
        key: list(values) if isinstance(values, list) else []
        for key, values in history.items()
    }
    if 'total' not in normalized and normalized.get('train_loss'):
        normalized['total'] = list(normalized['train_loss'])
    if 'val_total' not in normalized and normalized.get('val_loss'):
        normalized['val_total'] = list(normalized['val_loss'])
    if 'train_loss' not in normalized and normalized.get('total'):
        normalized['train_loss'] = list(normalized['total'])
    if 'val_loss' not in normalized and normalized.get('val_total'):
        normalized['val_loss'] = list(normalized['val_total'])
    return normalized


def load_history_file(run_dir: str) -> dict:
    """!
    @brief Load loss_history.json from a run directory if present.
    """
    path = os.path.join(run_dir, 'loss_history.json')
    if not os.path.isfile(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return normalize_history(json.load(f))


def best_metric_from_history(history: dict, monitor_key: str) -> Optional[float]:
    """!
    @brief Return the best finite monitored metric from resumed history.
    """
    values = history.get(monitor_key, [])
    finite_values = [float(value) for value in values if np.isfinite(value)]
    return min(finite_values) if finite_values else None


def historical_patience_counter(history: dict, monitor_key: str, best_metric: Optional[float]) -> int:
    """!
    @brief Rebuild the early-stopping counter from historical validation losses.
    """
    if best_metric is None:
        return 0
    values = history.get(monitor_key, [])
    counter = 0
    for value in reversed(values):
        if not np.isfinite(value):
            counter += 1
        elif float(value) > best_metric:
            counter += 1
        else:
            break
    return counter


def save_yaml(data: dict, path: str) -> None:
    """!
    @brief Save a YAML file without reordering keys.
    """
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_history(history: dict, run_dir: str) -> None:
    """!
    @brief Save loss history as JSON in a run directory.
    """
    with open(os.path.join(run_dir, 'loss_history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)


def _plot_keys(history: dict, keys: Iterable[str], path: str, title: str) -> None:
    """!
    @brief Plot a selected group of history keys when at least one is present.
    """
    has_any = any(key in history and history[key] for key in keys)
    if not has_any:
        return
    plt.figure(figsize=(12, 6))
    for key in keys:
        values = history.get(key, [])
        if values:
            plt.plot(values, label=key.replace('_', ' ').title())
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()


def plot_history(history: dict, run_dir: str) -> None:
    """!
    @brief Write standard training and validation loss plots.
    """
    train_keys = [
        key for key in history
        if history.get(key)
        and not key.startswith('val_')
        and key not in {'train_loss', 'val_loss'}
    ]
    val_keys = [key for key in history if key.startswith('val_') and history.get(key)]
    _plot_keys(history, train_keys, os.path.join(run_dir, 'losses_train.png'), 'Training Losses')
    _plot_keys(history, val_keys, os.path.join(run_dir, 'losses_val.png'), 'Validation Losses')

    total = history.get('total') or history.get('train_loss') or []
    val_total = history.get('val_total') or history.get('val_loss') or []
    if total or val_total:
        plt.figure(figsize=(10, 5))
        if total:
            plt.plot(range(1, len(total) + 1), total, label='Training Loss')
        if val_total:
            plt.plot(range(1, len(val_total) + 1), val_total, label='Validation Loss')
            finite = [(idx, value) for idx, value in enumerate(val_total, start=1) if np.isfinite(value)]
            if finite:
                best_epoch, best_value = min(finite, key=lambda item: item[1])
                plt.scatter(best_epoch, best_value, c='red', marker='*', s=120, label=f'Best ({best_value:.4f})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(run_dir, 'loss_plot.png'))
        plt.close()


def estimate_preprocessed_variance(
    dataloader: DataLoader,
    batch_preprocessor: torch.nn.Module,
    device: torch.device,
    max_samples: int = 1000,
) -> float:
    """!
    @brief Estimate variance after GPU preprocessing for VQ-VAE loss scaling.

    Raw-audio training computes Mel spectrograms inside the batch preprocessor,
    so the variance must be measured after that transformation rather than on
    waveform samples.
    """
    if max_samples < 1:
        raise ValueError(f'max_samples must be >= 1, got {max_samples}')
    was_training = batch_preprocessor.training
    batch_preprocessor.eval()
    values = []
    total_samples = 0
    with torch.no_grad():
        for raw_batch in tqdm(dataloader, desc='Estimating transformed data variance'):
            batch = move_to_device(raw_batch, device)
            batch = batch_preprocessor(batch, augment=False)
            if not torch.isfinite(batch).all():
                continue
            remaining = max_samples - total_samples
            if remaining <= 0:
                break
            batch = batch[:remaining]
            values.append(batch.detach().float().cpu().reshape(batch.shape[0], -1))
            total_samples += batch.shape[0]
            if total_samples >= max_samples:
                break
    if was_training:
        batch_preprocessor.train()
    if not values:
        raise ValueError('Could not estimate data variance: no finite preprocessed batches were found.')
    variance = float(torch.var(torch.cat(values, dim=0), unbiased=False).item())
    print(f'Estimated transformed data variance ({total_samples} samples max): {variance:.6f}')
    return max(variance, 1e-12)


def collect_callback_samples(
    dataset,
    batch_preprocessor,
    collate_fn,
    device: torch.device,
    sample_count: int = 4,
):
    """!
    @brief Collect denormalization-ready samples for reconstruction callbacks.
    """
    sample_count = min(int(sample_count), len(dataset))
    if sample_count <= 0:
        return np.empty((0, 0, 0, 1), dtype=np.float32), []

    if batch_preprocessor is not None:
        if collate_fn is None:
            raise ValueError('A collate_fn is required when collecting preprocessed callback samples.')
        raw_items = [dataset[i] for i in range(sample_count)]
        batch = move_to_device(collate_fn(raw_items), device)
        was_training = batch_preprocessor.training
        batch_preprocessor.eval()
        with torch.no_grad():
            output = batch_preprocessor(batch, augment=False, return_min_max=True)
        if was_training:
            batch_preprocessor.train()
        samples, sample_min_max = output
        samples = samples.detach().cpu().permute(0, 2, 3, 1).numpy()
        return samples, sample_min_max

    samples = []
    for idx in range(sample_count):
        item = dataset[idx]
        if isinstance(item, (tuple, list)):
            item = item[0]
        if torch.is_tensor(item):
            item = item.detach().cpu()
        if item.ndim == 3:
            item = item.permute(1, 2, 0).numpy()
        elif item.ndim == 2:
            item = item.numpy()[..., np.newaxis]
        samples.append(item.astype(np.float32, copy=False))
    return np.stack(samples, axis=0), None


def make_sample_generator(
    model,
    data_bundle: DataBundle,
    run_dir: str,
    device: torch.device,
    audio_settings: dict,
    sample_count: int = 4,
):
    """!
    @brief Build the shared SampleGenerator callback for VQ-VAE adapters.
    """
    sample_source = data_bundle.val_dataset or data_bundle.train_dataset
    if sample_source is None:
        return None

    samples, sample_min_max = collect_callback_samples(
        sample_source,
        data_bundle.batch_preprocessor,
        data_bundle.collate_fn,
        device,
        sample_count=sample_count,
    )
    if sample_min_max is None:
        sample_min_max = [{'min': 0.0, 'max': 1.0} for _ in range(len(samples))]

    return SampleGenerator(
        model,
        samples,
        sample_min_max,
        run_dir,
        device,
        spectrogram_type=audio_settings.get('spectrogram_type', 'linear'),
        hop_length=int(audio_settings.get('hop_length', 256)),
        sample_rate=int(audio_settings.get('sample_rate', 22050)),
        n_fft=int(audio_settings.get('frame_size', 2048)),
        n_mels=int(audio_settings.get('n_mels', 256)),
    )
