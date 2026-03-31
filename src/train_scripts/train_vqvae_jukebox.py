from datetime import datetime
import argparse
import torch
import numpy as np
import os
import json
import yaml
import sys
import pickle
import gc
import glob
import torch.nn as nn

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling.torch.jukebox_vq_vae import JukeboxVQVAE
from generation.generate import *
from utils import load_config
from train_scripts.train_vqvae_utils import train_vqvae_jukebox
from datasets.spectrogram_dataset import LazySpectrogramDataset

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def _extract_best_metric(history: dict):
    if not history:
        return None
    if 'val_total' in history and history['val_total']:
        finite_vals = [float(v) for v in history['val_total'] if np.isfinite(v)]
        if finite_vals:
            return min(finite_vals)
    if 'total' in history and history['total']:
        finite_vals = [float(v) for v in history['total'] if np.isfinite(v)]
        if finite_vals:
            return min(finite_vals)
    return None


def _load_resume_artifacts(pretrained_weights_path: str):
    if not os.path.isfile(pretrained_weights_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_weights_path}")

    checkpoint = torch.load(pretrained_weights_path, map_location='cpu', weights_only=False)
    resume_history = checkpoint.get('history', {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(resume_history, dict):
        resume_history = {}

    history_path = os.path.join(os.path.dirname(pretrained_weights_path), 'loss_history.json')
    if os.path.isfile(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            file_history = json.load(f)
        if isinstance(file_history, dict):
            resume_history = file_history

    best_metric = _extract_best_metric(resume_history)
    if best_metric is None and isinstance(checkpoint, dict) and 'metric_value' in checkpoint:
        best_metric = float(checkpoint['metric_value'])
    if best_metric is None and isinstance(checkpoint, dict) and 'loss' in checkpoint:
        best_metric = float(checkpoint['loss'])

    return resume_history, best_metric

if __name__ == "__main__":
    # Optional: faster matmul on Ampere+ GPUs
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Capability:", torch.cuda.get_device_capability(0))
        print("CUDA memory allocated (MB):", round(torch.cuda.memory_allocated(0)/1024**2, 2))

    parser = argparse.ArgumentParser(description="Train Jukebox-style VQ-VAE level model")
    parser.add_argument(
        "--level",
        type=str,
        choices=["bottom", "middle", "top"],
        default=None,
        help="Override config model.selected_level.",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: ignoring unrecognized arguments: {unknown}")

    # Load configuration
    config_path = "./config/config_jukebox.yaml"
    config = load_config(config_path)

    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    epochs = config['training']['epochs']
    early_stopping_patience = config['training'].get('early_stopping_patience', 20)
    num_workers = config['training'].get('num_workers', 4) # Default to 4 for lazy loading
    pin_memory = config['training'].get('pin_memory', True) # Default to True for speed
    persist_workers = bool(config['training'].get('persist_workers', None))
    prefetch_factor = int(config['training'].get('prefetch_factor', None))
    spectrograms_path = config['dataset']['processed_path']
    target_time_frames = int(config['dataset'].get('target_time_frames', 256))
    min_max_values_path = config['dataset']['min_max_values_path']
    model_save_dir = config['training']['save_dir']
    model_name = config['model']['name']
    model_cfg = config['model']
    retrain = bool(config['training'].get('retrain', False))
    pretrained_weights_path = config['training'].get('pretrained_weights_path')

    print(f"Configuration loaded. Model: {model_name}, Save Dir: {model_save_dir}")
    print(f"Training parameters: batch_size={batch_size}, learning_rate={learning_rate}, epochs={epochs}, early_stopping_patience={early_stopping_patience}")
    print(f"Data loading parameters: num_workers={num_workers}, pin_memory={pin_memory}, persist_workers={persist_workers}, prefetch_factor={prefetch_factor}")
    print(f"Dataset target_time_frames={target_time_frames}")

    selected_level = args.level or model_cfg.get('selected_level', 'bottom')
    selected_level = str(selected_level).lower()

    level_profiles = model_cfg.get('level_profiles')
    if level_profiles is None:
        legacy_levels = model_cfg.get('levels')
        legacy_res_layers = model_cfg.get('num_residual_layers', 4)
        if legacy_levels is None:
            raise ValueError("Missing model.level_profiles and legacy model.levels in config.")
        level_profiles = {
            selected_level: {
                'levels': legacy_levels,
                'num_residual_layers': legacy_res_layers,
            }
        }

    if selected_level not in level_profiles:
        available = ', '.join(level_profiles.keys())
        raise ValueError(f"Invalid selected level '{selected_level}'. Available levels: {available}")

    selected_profile = level_profiles[selected_level]
    levels = selected_profile.get('levels')
    num_residual_layers = selected_profile.get('num_residual_layers', 4)
    if levels is None:
        raise ValueError(f"Missing 'levels' for profile '{selected_level}' in model.level_profiles")

    current_datetime = datetime.now()
    formatted_time = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    # structure: model_save_dir / formatted_time / model.pth
    run_dir = os.path.join(model_save_dir, f"{model_name}_{selected_level}", formatted_time)
    os.makedirs(run_dir, exist_ok=True)

    model_file_path = os.path.join(run_dir, "model.pth")
    config_file_path = os.path.join(run_dir, "config.yaml")

    print(f"Training configuration loaded from {config_path}")
    print(f"Selected Jukebox level: {selected_level} (levels={levels}, residual_layers={num_residual_layers})")
    print(f"Model will be saved to: {model_file_path}")

    resume_history = {}
    initial_best_metric = None
    if retrain:
        if not pretrained_weights_path:
            raise ValueError("training.retrain is true but training.pretrained_weights_path is empty.")
        resume_history, initial_best_metric = _load_resume_artifacts(pretrained_weights_path)
        print(f"Retraining enabled from checkpoint: {pretrained_weights_path}")
        if initial_best_metric is not None:
            print(f"Baseline best metric from previous training: {initial_best_metric:.6f}")
        else:
            print("Baseline best metric unavailable from previous training history.")

    # Save the config used for this training run immediately
    config_to_save = dict(config)
    config_to_save['training'] = dict(config['training'])
    config_to_save['training']['retrain'] = retrain
    config_to_save['training']['pretrained_weights_path'] = pretrained_weights_path
    config_to_save['model'] = dict(config['model'])
    config_to_save['model']['selected_level'] = selected_level
    config_to_save['dataset'] = dict(config['dataset'])

    with open(config_file_path, 'w') as f:
        yaml.dump(config_to_save, f)
    
    # --- LAZY LOADING IMPLEMENTATION ---
    print(f"Scanning for spectrogram files in {spectrograms_path}...")
    all_file_paths = glob.glob(os.path.join(spectrograms_path, "**/*.npy"), recursive=True)
    
    if not all_file_paths:
        raise FileNotFoundError(f"No .npy files found in {spectrograms_path}")
    
    print(f"Found {len(all_file_paths)} files. Creating lazy datasets...")

    # Data variance calculation on a small subset (100 files) to save time/RAM
    subset_paths = all_file_paths[:min(100, len(all_file_paths))]
    subset_data = np.stack([np.load(p) for p in subset_paths])
    data_variance = np.var(subset_data)
    del subset_data
    gc.collect()

    # Split into train/val
    np.random.shuffle(all_file_paths)
    validation_split = config['training'].get('validation_split', 0.2)
    
    if validation_split > 0:
        num_val = int(len(all_file_paths) * validation_split)
        num_train = len(all_file_paths) - num_val
        
        train_paths = all_file_paths[:num_train]
        val_paths = all_file_paths[num_train:]
        
        # Instantiate Lazy Datasets
        x_train = LazySpectrogramDataset(train_paths, target_time_frames=target_time_frames)
        x_val = LazySpectrogramDataset(val_paths, target_time_frames=target_time_frames)
        
        train_file_paths = train_paths
        val_file_paths = val_paths
        
        print(f"Data split: {len(x_train)} training, {len(x_val)} validation samples.")
    else:
        x_train = LazySpectrogramDataset(all_file_paths, target_time_frames=target_time_frames)
        train_file_paths = all_file_paths
        x_val = None
        val_file_paths = None

    # Load min_max_values
    with open(min_max_values_path, "rb") as f:
        min_max_values = pickle.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    activation_name = str(model_cfg.get('activation', '')).lower()
    activation_layer = nn.Sigmoid() if activation_name == 'sigmoid' else None

    # Assert model config has all paramters needed for model initialization
    required_params = ['input_channels', 'hidden_dim', 'num_embeddings', 'embedding_dim', 'beta', 'conv_type', 'dilation_growth_rate', 'channel_growth', 'ema_decay', 'epsilon', 'restart_threshold']
    missing_params = [p for p in required_params if p not in model_cfg]
    if missing_params:
        raise ValueError(f"Missing required model config parameters: {', '.join(missing_params)}")

    jukebox_model = JukeboxVQVAE(
        input_channels=model_cfg['input_channels'],
        hidden_dim=model_cfg['hidden_dim'],
        levels=levels,
        num_residual_layers=num_residual_layers,
        num_embeddings=model_cfg.get('num_embeddings'),
        embedding_dim=model_cfg.get('embedding_dim'),
        beta=model_cfg.get('beta'),
        conv_type=model_cfg.get('conv_type'),
        activation_layer=activation_layer,
        dilation_growth_rate=model_cfg.get('dilation_growth_rate'),
        channel_growth=model_cfg.get('channel_growth'),
        ema_decay=model_cfg.get('ema_decay'),
        epsilon=model_cfg.get('epsilon'),
        restart_threshold=model_cfg.get('restart_threshold'),
    ).to(device)

    train_vqvae_jukebox(
        model=jukebox_model,
        x_train=x_train,
        train_file_paths=train_file_paths,
        min_max_values=min_max_values,
        device=device,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        save_path=model_file_path,
        data_variance=data_variance,
        early_stopping_patience=early_stopping_patience,
        x_val=x_val,
        val_file_paths=val_file_paths,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persist_workers=persist_workers,
        prefetch_factor=prefetch_factor,
        resume_checkpoint_path=pretrained_weights_path if retrain else None,
        resume_history=resume_history,
        initial_best_metric=initial_best_metric,
    )

    best_model_path = os.path.join(run_dir, 'best_model.pth')
    if retrain:
        if os.path.isfile(best_model_path):
            print("Retraining improved over baseline: new best model saved.")
        else:
            print("Retraining did not beat baseline best metric in this run.")
    print("Training completed.")