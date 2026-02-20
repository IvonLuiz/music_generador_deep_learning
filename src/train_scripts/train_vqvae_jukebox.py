from datetime import datetime
import argparse
import torch
import numpy as np
import os
import yaml
import sys
import pickle
import gc
import torch.nn as nn

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling.torch.jukebox_vq_vae import JukeboxVQVAE
from generation.generate import *
from utils import load_maestro, load_config
from train_scripts.train_vqvae2_utils import train_vqvae_hierarchical
from processing.preprocess_audio import TARGET_TIME_FRAMES, MIN_MAX_VALUES_SAVE_DIR
from datasets.spectrogram_dataset import MmapSpectrogramDataset

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class JukeboxHierarchicalAdapter(nn.Module):
    """
    Adapter to reuse train_vqvae_hierarchical utility without modifying it.
    It maps single-level JukeboxVQVAE outputs to the expected 2-level VQ loss structure.
    """
    def __init__(self, model: JukeboxVQVAE):
        super().__init__()
        self.model = model

    def forward(self, x):
        x_recon, total_vq_loss, vq_details = self.model(x)
        vq_loss, codebook_loss, commitment_loss = vq_details[0]
        zero = torch.zeros_like(vq_loss)
        return x_recon, total_vq_loss, [
            (vq_loss, codebook_loss, commitment_loss),
            (zero, zero, zero),
        ]

    def reconstruct(self, x):
        return self.model.reconstruct(x)

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
    args = parser.parse_args()

    # Load configuration
    config_path = "./config/config_jukebox.yaml"
    config = load_config(config_path)

    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    epochs = config['training']['epochs']
    early_stopping_patience = config['training'].get('early_stopping_patience', 20)
    num_workers = config['training'].get('num_workers', 0)
    pin_memory = config['training'].get('pin_memory', False)
    spectrograms_path = config['dataset']['processed_path']
    model_save_dir = config['training']['save_dir']
    model_name = config['model']['name']
    model_cfg = config['model']

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

    # Save the config used for this training run immediately
    with open(config_file_path, 'w') as f:
        yaml.dump(config, f)
    
    # Load training data to compute variance
    x_all, file_paths_all = load_maestro(spectrograms_path, TARGET_TIME_FRAMES, debug_print=False)
    data_variance = np.var(x_all)

    # Split into train/val manually to save memory
    validation_split = config['training'].get('validation_split', 0.0)
    if validation_split > 0:
        num_samples = len(x_all)
        num_val = int(num_samples * validation_split)
        num_train = num_samples - num_val
        
        # Shuffle indices
        indices = np.random.permutation(num_samples)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]
        
        # Use MmapSpectrogramDataset to avoid loading data into RAM
        x_train = MmapSpectrogramDataset(x_all, train_indices)
        x_val = MmapSpectrogramDataset(x_all, val_indices)
        
        # Handle paths
        file_paths_all = np.array(file_paths_all)
        train_file_paths = file_paths_all[train_indices].tolist()
        val_file_paths = file_paths_all[val_indices].tolist()
        
        print(f"Data split: {len(x_train)} training, {len(x_val)} validation samples.")
        
        # We keep x_all referenced by datasets, so we don't delete it explicitly, 
        # but we can delete file_paths_all
        del file_paths_all
        gc.collect()
    else:
        x_train = MmapSpectrogramDataset(x_all)
        train_file_paths = file_paths_all
        x_val = None
        val_file_paths = None

    # Load min_max_values
    min_max_values_path = os.path.join(MIN_MAX_VALUES_SAVE_DIR, "min_max_values.pkl")
    with open(min_max_values_path, "rb") as f:
        min_max_values = pickle.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    activation_name = str(model_cfg.get('activation', '')).lower()
    activation_layer = nn.Sigmoid() if activation_name == 'sigmoid' else None

    jukebox_model = JukeboxVQVAE(
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

    train_model = JukeboxHierarchicalAdapter(jukebox_model)
    
    # Train the VQ-VAE Hierarchical model
    train_vqvae_hierarchical(
        model=train_model,
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
    )
    print("Training completed.")