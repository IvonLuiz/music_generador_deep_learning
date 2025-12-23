import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend to avoid thread issues
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import torch.nn.functional as F

from modeling.torch.vq_vae_hierarchical import VQ_VAE_Hierarchical
from datasets.spectrogram_dataset import SpectrogramDataset, MmapSpectrogramDataset
from processing.preprocess_audio import HOP_LENGTH, SAMPLE_RATE
from utils import find_min_max_for_path
from callbacks import EarlyStopping, ModelCheckpoint, LossPlotter, SampleGenerator


def train_vqvae_hierarchical(model: VQ_VAE_Hierarchical,
                             x_train: np.ndarray,
                             train_file_paths: list,
                             min_max_values: dict,
                             data_variance: float,
                             batch_size: int,
                             epochs: int,
                             learning_rate: float,
                             save_path: str,
                             device: torch.device,
                             amp: bool = True,
                             x_val: np.ndarray = None,
                             val_file_paths: list = None):
    """
    Train a VQ-VAE Hierarchical model.

    Args:
        model (VQ_VAE_Hierarchical): The VQ-VAE Hierarchical model to train.
        x_train (np.ndarray): Training spectrogram data.
        train_file_paths (list): List of file paths corresponding to x_train.
        min_max_values (dict): Dictionary of min/max values for denormalization.
        data_variance (float): Variance of the training data for loss scaling.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        save_path (str): Path to save the trained model.
        device (torch.device): Device to run the training on (CPU or GPU).
        amp (bool, optional): Whether to use automatic mixed precision (AMP) during training. Defaults to True.
        x_val (np.ndarray, optional): Validation spectrogram data.
        val_file_paths (list, optional): List of file paths corresponding to x_val.
    """
    model.to(device)
    
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
                early_stopping = EarlyStopping(patience=20, verbose=True)
        else:
            # Assume x_val is a Dataset
            print(f"Training with {len(x_train)} samples and validating with {len(x_val)} samples.")
            val_dataset = x_val
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            early_stopping = EarlyStopping(patience=20, verbose=True)
    
    if val_dataloader is None:
        print(f"Using all {len(x_train)} samples for training (no validation set provided).")

    if isinstance(x_train, (np.ndarray, list)):
        dataset = SpectrogramDataset(x_train)
    else:
        dataset = x_train
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner for potential speedup
    scaler = GradScaler(enabled=amp and device.type == 'cuda')  # For mixed precision training

    print("Model will be saved to :", save_path)
    
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
                # Fetch samples from dataset and convert to numpy (N, H, W, 1)
                samples = []
                for i in range(4):
                    # Dataset returns (1, H, W) tensor
                    s = val_dataset[i]
                    s = s.permute(1, 2, 0).numpy() # (H, W, 1)
                    samples.append(s)
                samples = np.stack(samples)
            else:
                samples = x_val[:4]
            sample_paths = val_file_paths[:4]
        else:
            if isinstance(dataset, Dataset):
                samples = []
                for i in range(4):
                    s = dataset[i]
                    s = s.permute(1, 2, 0).numpy()
                    samples.append(s)
                samples = np.stack(samples)
            else:
                samples = x_train[:4]
            sample_paths = train_file_paths[:4]
        
        spectrograms_dir = os.path.dirname(sample_paths[0])
        sample_min_max = []
        for fp in sample_paths:
            mm = find_min_max_for_path(fp, min_max_values, spectrograms_dir)
            if mm is None:
                print(f"Warning: Could not find min/max for {fp}. Using default 0-1.")
                mm = {"min": 0.0, "max": 1.0}
            sample_min_max.append(mm)
            
        sample_generator = SampleGenerator(model, samples, sample_min_max, os.path.dirname(save_path), device)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad(set_to_none=True)
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_vq_loss_top = 0.0
        epoch_vq_loss_bottom = 0.0
        total_samples = 0
        
        for batch in progress_bar:
            batch = batch.to(device)

            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=scaler.is_enabled()):
                reconstructions, total_vq_loss, vq_losses_details = model(batch)
                
                # Unpack losses
                # vq_losses_details is [(vq_loss_top, codebook_loss_top, commitment_loss_top), (vq_loss_bottom, ...)]
                (vq_loss_top, codebook_loss_top, commitment_loss_top) = vq_losses_details[0]
                (vq_loss_bottom, codebook_loss_bottom, commitment_loss_bottom) = vq_losses_details[1]
                
                recon_loss = F.mse_loss(reconstructions, batch) / (2 * data_variance)
                loss = recon_loss + total_vq_loss

            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            batch_size_current = batch.size(0)
            epoch_loss += loss.item() * batch_size_current
            epoch_recon_loss += recon_loss.item() * batch_size_current
            epoch_vq_loss_top += vq_loss_top.item() * batch_size_current
            epoch_vq_loss_bottom += vq_loss_bottom.item() * batch_size_current
            total_samples += batch_size_current

            progress_bar.set_postfix(loss=epoch_loss / total_samples)

        # Calculate average losses for the epoch
        avg_epoch_loss = epoch_loss / len(dataset)
        avg_recon_loss = epoch_recon_loss / len(dataset)
        avg_vq_loss_top = epoch_vq_loss_top / len(dataset)
        avg_vq_loss_bottom = epoch_vq_loss_bottom / len(dataset)

        # Update Loss Plotter
        epoch_metrics = {
            'total': avg_epoch_loss,
            'reconstruction_loss': avg_recon_loss,
            'vq_loss_top': avg_vq_loss_top,
            'vq_loss_bottom': avg_vq_loss_bottom
        }

        # Validation Loop
        val_loss_str = ""
        avg_val_loss = None
        
        if val_dataloader:
            model.eval()
            val_epoch_loss = 0.0
            val_epoch_recon_loss = 0.0
            val_epoch_vq_loss_top = 0.0
            val_epoch_vq_loss_bottom = 0.0
            val_total_samples = 0
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    batch = batch.to(device)
                    reconstructions, total_vq_loss, vq_losses_details = model(batch)
                    
                    (vq_loss_top, _, _) = vq_losses_details[0]
                    (vq_loss_bottom, _, _) = vq_losses_details[1]
                    
                    recon_loss = F.mse_loss(reconstructions, batch) / (2 * data_variance)
                    loss = recon_loss + total_vq_loss
                    
                    batch_size_current = batch.size(0)
                    val_epoch_loss += loss.item() * batch_size_current
                    val_epoch_recon_loss += recon_loss.item() * batch_size_current
                    val_epoch_vq_loss_top += vq_loss_top.item() * batch_size_current
                    val_epoch_vq_loss_bottom += vq_loss_bottom.item() * batch_size_current
                    val_total_samples += batch_size_current
            
            avg_val_loss = val_epoch_loss / len(val_dataset)
            avg_val_recon_loss = val_epoch_recon_loss / len(val_dataset)
            avg_val_vq_loss_top = val_epoch_vq_loss_top / len(val_dataset)
            avg_val_vq_loss_bottom = val_epoch_vq_loss_bottom / len(val_dataset)
            
            epoch_metrics.update({
                'val_total': avg_val_loss,
                'val_reconstruction_loss': avg_val_recon_loss,
                'val_vq_loss_top': avg_val_vq_loss_top,
                'val_vq_loss_bottom': avg_val_vq_loss_bottom
            })
            
            val_loss_str = f", Val Loss: {avg_val_loss:.4f}"

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}, Recon: {avg_recon_loss:.4f}, VQ Top: {avg_vq_loss_top:.4f}, VQ Bottom: {avg_vq_loss_bottom:.4f}{val_loss_str}")

        # Callbacks Step
        if loss_plotter:
            loss_plotter.update(epoch_metrics)
            loss_plotter.plot()
            
        if model_checkpoint:
            # Use validation loss if available, else training loss
            metric_to_monitor = avg_val_loss if avg_val_loss is not None else avg_epoch_loss
            model_checkpoint.step(epoch, avg_epoch_loss, metric_value=metric_to_monitor)
            
        if sample_generator:
            sample_generator.step(epoch)

        # Early Stopping Check
        if early_stopping and avg_val_loss is not None:
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return model
