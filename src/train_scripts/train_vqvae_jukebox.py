from datetime import datetime
import argparse
import torch
import numpy as np
import os
import random
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
from utils import load_config, compute_dataset_variance, compute_small_sample_variance
from train_scripts.train_vqvae_utils import train_vqvae_jukebox
from datasets.spectrogram_dataset import LazySpectrogramDataset
from resume_utils import load_resume_artifacts

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def set_global_seed(seed: int) -> None:
    """Set global RNG seeds without forcing deterministic kernels (keeps training fast)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

    # Determine selected level first
    model_cfg = config['model']
    selected_level = args.level or model_cfg.get('selected_level', 'bottom')
    selected_level = str(selected_level).lower()

    # Merge general training config with level-specific config
    train_general_cfg = config['training']
    train_level_cfg = train_general_cfg.get(selected_level, {})
    # Filter out level-specific keys and merge
    train_cfg = {k: v for k, v in train_general_cfg.items() if k not in ['bottom', 'middle', 'top']}
    train_cfg.update(train_level_cfg)
    seed = int(train_cfg.get('seed', 42))
    set_global_seed(seed)

    # Extract parameters
    batch_size = train_cfg.get('batch_size')
    grad_accum_steps = train_cfg.get('gradient_accumulation_steps', 1)  # Default to 1
    learning_rate = train_cfg.get('learning_rate')
    epochs = train_cfg.get('epochs')
    validation_split = train_cfg.get('validation_split', 0.2)
    early_stopping_patience = train_cfg.get('early_stopping_patience', 20)
    num_workers = train_cfg.get('num_workers', 4)
    pin_memory = train_cfg.get('pin_memory', True)
    persist_workers_cfg = train_cfg.get('persist_workers', True)
    persist_workers = bool(persist_workers_cfg) if persist_workers_cfg is not None else True
    prefetch_factor_cfg = train_cfg.get('prefetch_factor', 4)
    prefetch_factor = int(prefetch_factor_cfg) if prefetch_factor_cfg is not None else None

    # Dataset and model paths
    dataset_cfg = config['dataset']
    spectrograms_path = dataset_cfg['processed_path']
    target_time_frames = int(dataset_cfg.get('target_time_frames', 256))
    min_max_values_path = dataset_cfg['min_max_values_path']
    sample_rate = int(dataset_cfg.get('sample_rate', 22050))
    hop_length = int(dataset_cfg.get('hop_length', 256))
    frame_size = int(dataset_cfg.get('frame_size', 512))
    spectrogram_type_cfg = dataset_cfg.get('spectrogram_type')
    if spectrogram_type_cfg is None:
        spectrogram_type = 'mel' if 'mel' in str(spectrograms_path).lower() else 'linear'
    else:
        spectrogram_type = str(spectrogram_type_cfg).strip().lower()
    n_mels = int(dataset_cfg.get('n_mels', 256))
    model_save_dir = train_cfg['save_dir']
    model_name = model_cfg['name']
    retrain = bool(train_cfg.get('retrain', False))
    pretrained_weights_path = train_cfg.get('pretrained_weights_path')

    print(f"Configuration loaded. Model: {model_name}, Level: {selected_level}, Save Dir: {model_save_dir}")
    print(f"Training parameters: batch_size={batch_size}, grad_accum_steps={grad_accum_steps}, learning_rate={learning_rate}, epochs={epochs}, early_stopping_patience={early_stopping_patience}")
    print(f"Reproducibility seed: {seed}")
    print(f"Data loading parameters: num_workers={num_workers}, pin_memory={pin_memory}, persist_workers={persist_workers}, prefetch_factor={prefetch_factor}")
    print(f"Dataset target_time_frames={target_time_frames}, validation_split={validation_split}")
    print(f"Audio inversion settings: spectrogram_type={spectrogram_type}, sample_rate={sample_rate}, hop_length={hop_length}, frame_size={frame_size}, n_mels={n_mels}")

    # Individual level parameters
    level_profiles = model_cfg.get('level_profiles')
    if level_profiles is None:
        raise ValueError("model.level_profiles is missing from config. It should define parameters for each level (bottom, middle, top).")
    if selected_level not in level_profiles:
        available = ', '.join(level_profiles.keys())
        raise ValueError(f"Invalid selected level '{selected_level}'. Available levels: {available}")

    selected_profile = level_profiles[selected_level]
    hidden_dim = selected_profile.get('hidden_dim')
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
        resume_history, initial_best_metric, _ = load_resume_artifacts(pretrained_weights_path, val_key='val_total', train_key='total')
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
    all_file_paths = sorted(all_file_paths)
    
    if not all_file_paths:
        raise FileNotFoundError(f"No .npy files found in {spectrograms_path}")
    
    print(f"Found {len(all_file_paths)} files. Creating lazy datasets...")

    # Data variance calculation
    data_variance = compute_dataset_variance(all_file_paths)
    print(f"Computed dataset variance: {data_variance:.6f}")
    data_variance_small_sample = compute_small_sample_variance(all_file_paths, samples=500)
    print(f"Computed small sample variance (500 samples): {data_variance_small_sample:.6f}")
    gc.collect()

    # Split into train/val
    split_rng = np.random.default_rng(seed)
    split_rng.shuffle(all_file_paths)
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
    required_params = ['input_channels', 'num_embeddings', 'embedding_dim', 'beta', 'conv_type', 'dilation_growth_rate', 'channel_growth', 'ema_decay', 'epsilon', 'restart_threshold']
    missing_params = [p for p in required_params if p not in model_cfg]
    if missing_params:
        raise ValueError(f"Missing required model config parameters: {', '.join(missing_params)}")

    jukebox_model = JukeboxVQVAE(
        input_channels=model_cfg['input_channels'],
        hidden_dim=hidden_dim,
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
        grad_accum_steps=grad_accum_steps,
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
        spectrogram_type=spectrogram_type,
        sample_rate=sample_rate,
        hop_length=hop_length,
        frame_size=frame_size,
        n_mels=n_mels,
        seed=seed,
    )

    best_model_path = os.path.join(run_dir, 'best_model.pth')
    if retrain:
        if os.path.isfile(best_model_path):
            print("Retraining improved over baseline: new best model saved.")
        else:
            print("Retraining did not beat baseline best metric in this run.")
    print("Training completed.")