import os
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

from processing.preprocess_audio import HOP_LENGTH
from generation.soundgenerator import SoundGenerator
import soundfile as sf

from modeling.torch.vq_vae import VQ_VAE, vqvae_loss
from datasets.spectrogram_dataset import SpectrogramDataset
from callbacks import EarlyStopping, ModelCheckpoint, LossPlotter, SampleGenerator


def train_vqvae(x_train: np.ndarray,
                input_shape: Tuple[int, int, int],
                conv_filters=(256, 128, 64, 32),
                conv_kernels=(3, 3, 3, 3),
                conv_strides=((2, 2), (2, 2), (2, 2), (2, 1)),
                embeddings_size=256,
                latent_space_dim=128,
                learning_rate=5e-4,
                batch_size=64,
                epochs=50,
                data_variance: float = 1.0,
                save_path: Optional[str] = None,
                early_stopping_patience: int = 20,
                amp: bool = True,
                grad_accum_steps: int = 1,
                max_grad_norm: Optional[float] = None,
                min_max_values: Optional[list] = None,
                x_val: Optional[np.ndarray] = None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = VQ_VAE(
        input_shape=input_shape,
        conv_filters=conv_filters,
        conv_kernels=conv_kernels,
        conv_strides=conv_strides,
        latent_space_dim=latent_space_dim,
        embeddings_size=embeddings_size,
    ).to(device)
    
    config = {
        'input_shape': input_shape,
        'conv_filters': conv_filters,
        'conv_kernels': conv_kernels,
        'conv_strides': conv_strides,
        'latent_space_dim': latent_space_dim,
        'embeddings_size': embeddings_size,
    }

    return train_model(
        model=model,
        x_train=x_train,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        data_variance=data_variance,
        save_path=save_path,
        early_stopping_patience=early_stopping_patience,
        amp=amp,
        grad_accum_steps=grad_accum_steps,
        max_grad_norm=max_grad_norm,
        model_config=config,
        min_max_values=min_max_values,
        x_val=x_val
    )


def train_model(model: VQ_VAE,
                x_train: np.ndarray,
                batch_size: int = 64,
                epochs: int = 50,
                learning_rate: float = 5e-4,
                data_variance: float = 1.0,
                early_stopping_patience: int = 20,
                save_path: Optional[str] = None,
                amp: bool = True,
                grad_accum_steps: int = 1,
                max_grad_norm: Optional[float] = None,
                model_config: Optional[dict] = None,
                min_max_values: Optional[list] = None,
                x_val: Optional[np.ndarray] = None):
    """
    Train an existing VQ-VAE model.
    
    Args:
        model: Pre-instantiated VQ_VAE model
        x_train: Training data of shape (N, H, W, 1)
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for Adam optimizer
        early_stopping_patience: Patience for early stopping
        data_variance: Data variance for loss calculation
        save_path: Optional path to save the model after training
        model_config: Optional dictionary with model configuration to save
        min_max_values: Optional list of min/max values for reconstruction visualization
        x_val: Optional validation data
    
    Returns:
        The trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = model.to(device)
    
    if isinstance(x_train, (np.ndarray, list)):
        ds = SpectrogramDataset(x_train)
    else:
        ds = x_train
        
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Setup Validation Data
    early_stopping = None
    val_dataloader = None
    val_dataset = None

    if x_val is not None:
        if isinstance(x_val, (np.ndarray, list)):
            if len(x_val) > 0:
                print(f"Training with {len(x_train)} samples and validating with {len(x_val)} samples.")
                val_dataset = SpectrogramDataset(x_val)
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
                early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
        else:
            # Assume x_val is a Dataset
            print(f"Training with {len(x_train)} samples and validating with {len(x_val)} samples.")
            val_dataset = x_val
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
    
    if val_dataloader is None:
        print(f"Using all {len(x_train)} samples for training (no validation set provided).")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    torch.backends.cudnn.benchmark = True
    scaler = GradScaler(enabled=(amp and device.type == 'cuda'))

    # Track losses for training progress
    train_losses_dict = {
        'total': [], 
        'codebook': [], 'commitment': [],
        'reconstruction': [], 'vq': []
    }
    
    print("Model will be saved to :", save_path) if save_path else None

    # Initialize Callbacks
    model_checkpoint = None
    loss_plotter = None
    sample_generator = None

    if save_path:
        model_checkpoint = ModelCheckpoint(save_path, model, optimizer, mode="min")
        loss_plotter = LossPlotter(save_path)
        
        # Prepare samples for visualization
        if val_dataloader:
            if isinstance(val_dataset, Dataset):
                samples = []
                for i in range(4):
                    s = val_dataset[i]
                    s = s.permute(1, 2, 0).numpy()
                    samples.append(s)
                samples = np.stack(samples)
            else:
                samples = x_val[:4]
        else:
            if isinstance(ds, Dataset):
                samples = []
                for i in range(4):
                    s = ds[i]
                    s = s.permute(1, 2, 0).numpy()
                    samples.append(s)
                samples = np.stack(samples)
            else:
                samples = x_train[:4]
        
        if min_max_values is None:
            # Create dummy min_max_values for visualization only (0-1 range)
            min_max_values = [{"min": 0.0, "max": 1.0} for _ in range(len(samples))]
        
        sample_generator = SampleGenerator(model, samples, min_max_values[:4], os.path.dirname(save_path), device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_codebook_loss, running_commitment_loss, running_recon_loss, running_vq_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        total_samples = 0
        optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(dl, desc=f"Epoch {epoch:03d}/{epochs}")
        for step, specs in enumerate(progress_bar, start=1):
            specs = specs.to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=scaler.is_enabled()):
                x_hat, _z, vq_loss, codebook_loss, commitment_loss = model(specs)
                loss_full, recon_loss  = vqvae_loss(specs, x_hat, vq_loss, variance=max(data_variance, 1e-6))
                loss = loss_full / grad_accum_steps
                
                # Accumulate individual losses for logging
                running_codebook_loss += codebook_loss.item() * specs.size(0)
                running_commitment_loss += commitment_loss.item() * specs.size(0)
                running_recon_loss += recon_loss.item() * specs.size(0)
                running_vq_loss += vq_loss.item() * specs.size(0)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if step % grad_accum_steps == 0:
                    if max_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if step % grad_accum_steps == 0:
                    if max_grad_norm is not None:
                        clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            batch_size_current = specs.size(0)
            running_loss += loss_full.item() * batch_size_current
            total_samples += batch_size_current
            progress_bar.set_postfix(loss=running_loss / total_samples)
        
        avg_loss = running_loss / len(ds)
        avg_codebook = running_codebook_loss / len(ds)
        avg_commitment = running_commitment_loss / len(ds)
        avg_recon = running_recon_loss / len(ds)
        avg_vq = running_vq_loss / len(ds)

        # Validation Loop
        val_loss_str = ""
        avg_val_loss = None
        
        if val_dataloader:
            model.eval()
            val_running_loss = 0.0
            val_running_recon_loss = 0.0
            val_running_vq_loss = 0.0
            val_total_samples = 0
            
            with torch.no_grad():
                for val_specs in val_dataloader:
                    val_specs = val_specs.to(device, non_blocking=True)
                    x_hat, _z, vq_loss, codebook_loss, commitment_loss = model(val_specs)
                    loss_full, recon_loss = vqvae_loss(val_specs, x_hat, vq_loss, variance=max(data_variance, 1e-6))
                    
                    batch_size_current = val_specs.size(0)
                    val_running_loss += loss_full.item() * batch_size_current
                    val_running_recon_loss += recon_loss.item() * batch_size_current
                    val_running_vq_loss += vq_loss.item() * batch_size_current
                    val_total_samples += batch_size_current
            
            if val_total_samples > 0:
                avg_val_loss = val_running_loss / val_total_samples
                avg_val_recon = val_running_recon_loss / val_total_samples
                avg_val_vq = val_running_vq_loss / val_total_samples
                val_loss_str = f"; val_loss {avg_val_loss:.6f}, val_recon {avg_val_recon:.6f}"

        print(f"Epoch {epoch:03d}/{epochs} - losses: running {avg_loss:.6f}; codebook {avg_codebook:.6f}, commitment {avg_commitment:.6f}, recon {avg_recon:.6f}, vq {avg_vq:.6f}{val_loss_str}")

        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Callbacks Step
        metrics = {
            'total': avg_loss,
            'codebook': avg_codebook,
            'commitment': avg_commitment,
            'reconstruction': avg_recon,
            'vq': avg_vq
        }
        
        if avg_val_loss is not None:
            metrics['val_total'] = avg_val_loss
            metrics['val_reconstruction'] = avg_val_recon
            metrics['val_vq'] = avg_val_vq

        if loss_plotter:
            loss_plotter.update(metrics)
            loss_plotter.plot()
            
        if model_checkpoint:
            metric_to_monitor = avg_val_loss if avg_val_loss is not None else avg_loss
            model_checkpoint.step(epoch, avg_loss, metric_value=metric_to_monitor)
            
        if sample_generator:
            sample_generator.step(epoch)

        # Early Stopping Check
        if early_stopping and avg_val_loss is not None:
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

    return model


def load_fsdd(path, add_channel_axis=True):
    """
    Loads spectrograms from a directory (recursively), returns (N, H, W, 1) and file paths.
    If add_channel_axis is True, ensures output shape is (N, H, W, 1) even if loaded .npy is (H, W).
    Compatible with both FSDD and MAESTRO spectrogram folders.
    """
    x_train = []
    file_paths = []

    for root, _, file_names in os.walk(path):
        for file_name in file_names:
            if file_name.endswith(".npy"):
                file_path = os.path.join(root, file_name)
                arr = np.load(file_path)
                # If shape is (H, W), add channel axis
                if add_channel_axis and arr.ndim == 2:
                    arr = arr[..., np.newaxis]
                x_train.append(arr)
                file_paths.append(file_path)

    x_train = np.array(x_train)
    return x_train, file_paths