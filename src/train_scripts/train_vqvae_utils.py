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
from modeling.torch.vq_vae import VQ_VAE
from modeling.torch.jukebox_vq_vae import JukeboxVQVAE
from datasets.spectrogram_dataset import SpectrogramDataset
from processing.preprocess_audio import HOP_LENGTH, SAMPLE_RATE
from utils import find_min_max_for_path
from callbacks import EarlyStopping, ModelCheckpoint, LossPlotter, SampleGenerator


EXPECTED_VQ_LEVELS = 2


def _split_two_level_vq_losses(vq_losses_details, context: str):
    if len(vq_losses_details) != EXPECTED_VQ_LEVELS:
        raise ValueError(
            f"{context}: expected exactly {EXPECTED_VQ_LEVELS} VQ levels for VQ-VAE2, "
            f"got {len(vq_losses_details)}"
        )
    top_vq_loss = vq_losses_details[0][0]
    bottom_vq_loss = vq_losses_details[1][0]
    return top_vq_loss, bottom_vq_loss


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
                             early_stopping_patience: int = 20,
                             amp: bool = True,
                             x_val: np.ndarray = None,
                             val_file_paths: list = None,
                             num_workers: int = 4,
                             pin_memory: bool = True,
                             persist_workers: bool = True,
                             prefetch_factor: int = 4,):
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
        early_stopping_patience (int, optional): Patience for early stopping. Defaults to 20.
        amp (bool, optional): Whether to use automatic mixed precision (AMP) during training. Defaults to True.
        x_val (np.ndarray, optional): Validation spectrogram data.
        val_file_paths (list, optional): List of file paths corresponding to x_val.
        num_workers (int, optional): Number of worker threads for data loading. Defaults to 4.
        pin_memory (bool, optional): Whether to pin memory for faster data transfer. Defaults to True.
        persist_workers (bool, optional): Whether to persist worker processes. Defaults to True.
        prefetch_factor (int, optional): Number of batches to prefetch. Defaults to 4.
    """
    model.to(device)
    
    # Setup Validation Data
    early_stopping = None
    val_dataloader = None
    val_dataset = None

    def _loader_kwargs():
        kwargs = {
            'num_workers': num_workers,
            'pin_memory': pin_memory,
        }
        if num_workers > 0:
            kwargs['persistent_workers'] = persist_workers
            if prefetch_factor is not None:
                kwargs['prefetch_factor'] = prefetch_factor
        return kwargs

    if x_val is not None:
        if isinstance(x_val, (np.ndarray, list)):
            if len(x_val) > 0:
                print(f"Training with {len(x_train)} samples and validating with {len(x_val)} samples.")
                val_dataset = SpectrogramDataset(x_val)
                val_dataloader = DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False,
                    **_loader_kwargs(),
                )
                early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
        else:
            # Assume x_val is a Dataset
            print(f"Training with {len(x_train)} samples and validating with {len(x_val)} samples.")
            val_dataset = x_val
            val_dataloader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                **_loader_kwargs(),
            )
            early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
    
    if val_dataloader is None:
        print(f"Using all {len(x_train)} samples for training (no validation set provided).")

    if isinstance(x_train, (np.ndarray, list)):
        dataset = SpectrogramDataset(x_train)
    else:
        dataset = x_train
        
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        **_loader_kwargs(),
    )

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
            sample_paths = val_file_paths[:4] if val_file_paths else None
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
            sample_paths = train_file_paths[:4] if train_file_paths else None
        
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
        epoch_vq_losses = [0.0, 0.0]  # [top, bottom]
        total_samples = 0
        skipped_non_finite_batches = 0
        
        for batch in progress_bar:
            batch = batch.to(device)

            if not torch.isfinite(batch).all():
                skipped_non_finite_batches += 1
                print(f"Warning: non-finite values in training batch at epoch {epoch+1}. Skipping batch.")
                continue

            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=scaler.is_enabled()):
                reconstructions, total_vq_loss, vq_losses_details = model(batch)
                top_vq_loss, bottom_vq_loss = _split_two_level_vq_losses(
                    vq_losses_details,
                    context='Training forward pass',
                )

                recon_loss = F.mse_loss(reconstructions, batch) / (2 * data_variance)
                loss = recon_loss + total_vq_loss

            if not torch.isfinite(loss):
                skipped_non_finite_batches += 1
                print(f"Warning: non-finite training loss at epoch {epoch+1}. Skipping optimizer step for this batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            batch_size_current = batch.size(0)
            epoch_loss += loss.item() * batch_size_current
            epoch_recon_loss += recon_loss.item() * batch_size_current
            epoch_vq_losses[0] += top_vq_loss.item() * batch_size_current
            epoch_vq_losses[1] += bottom_vq_loss.item() * batch_size_current
            total_samples += batch_size_current

            progress_bar.set_postfix(loss=epoch_loss / total_samples)

        if skipped_non_finite_batches > 0:
            print(f"Epoch {epoch+1}: skipped {skipped_non_finite_batches} non-finite training batches.")

        # Calculate average losses for the epoch
        if total_samples == 0:
            print(f"Epoch {epoch+1}: no valid training batches remained after filtering non-finite values. Stopping training.")
            break

        avg_epoch_loss = epoch_loss / total_samples
        avg_recon_loss = epoch_recon_loss / total_samples
        avg_vq_losses = [v / total_samples for v in epoch_vq_losses]

        # Update Loss Plotter
        epoch_metrics = {'total': avg_epoch_loss, 'reconstruction_loss': avg_recon_loss}
        epoch_metrics['vq_loss_top'] = avg_vq_losses[0]
        epoch_metrics['vq_loss_bottom'] = avg_vq_losses[1]

        # Validation Loop
        val_loss_str = ""
        avg_val_loss = None
        
        if val_dataloader:
            model.eval()
            val_epoch_loss = 0.0
            val_epoch_recon_loss = 0.0
            val_epoch_vq_losses = [0.0, 0.0]  # [top, bottom]
            val_total_samples = 0
            skipped_non_finite_val_batches = 0
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    batch = batch.to(device)

                    if not torch.isfinite(batch).all():
                        skipped_non_finite_val_batches += 1
                        print(f"Warning: non-finite values in validation batch at epoch {epoch+1}. Skipping batch.")
                        continue

                    reconstructions, total_vq_loss, vq_losses_details = model(batch)
                    top_vq_loss, bottom_vq_loss = _split_two_level_vq_losses(
                        vq_losses_details,
                        context='Validation forward pass',
                    )

                    recon_loss = F.mse_loss(reconstructions, batch) / (2 * data_variance)
                    loss = recon_loss + total_vq_loss

                    if not torch.isfinite(loss):
                        skipped_non_finite_val_batches += 1
                        print(f"Warning: non-finite validation loss at epoch {epoch+1}. Skipping batch.")
                        continue
                    
                    batch_size_current = batch.size(0)
                    val_epoch_loss += loss.item() * batch_size_current
                    val_epoch_recon_loss += recon_loss.item() * batch_size_current
                    val_epoch_vq_losses[0] += top_vq_loss.item() * batch_size_current
                    val_epoch_vq_losses[1] += bottom_vq_loss.item() * batch_size_current
                    val_total_samples += batch_size_current

            if skipped_non_finite_val_batches > 0:
                print(f"Epoch {epoch+1}: skipped {skipped_non_finite_val_batches} non-finite validation batches.")

            if val_total_samples == 0:
                avg_val_loss = float('nan')
                avg_val_vq_losses_val = [float('nan'), float('nan')]
            else:
                avg_val_loss = val_epoch_loss / val_total_samples
                avg_val_vq_losses_val = [v / val_total_samples for v in val_epoch_vq_losses]

            epoch_metrics['val_total'] = avg_val_loss
            epoch_metrics['val_reconstruction_loss'] = val_epoch_recon_loss / val_total_samples if val_total_samples else float('nan')
            epoch_metrics['val_vq_loss_top'] = avg_val_vq_losses_val[0]
            epoch_metrics['val_vq_loss_bottom'] = avg_val_vq_losses_val[1]

            val_loss_str = f", Val Loss: {avg_val_loss:.4f}"

        vq_str = f"VQ Top: {avg_vq_losses[0]:.4f}, VQ Bottom: {avg_vq_losses[1]:.4f}"
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}, Recon: {avg_recon_loss:.4f}, {vq_str}{val_loss_str}")

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
            if not np.isfinite(avg_val_loss):
                print("Validation loss is non-finite. Stopping training to prevent unstable checkpointing.")
                break
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return model


def train_vqvae_jukebox(model: JukeboxVQVAE,
                        x_train: np.ndarray,
                        train_file_paths: list,
                        min_max_values: dict,
                        data_variance: float,
                        batch_size: int,
                        epochs: int,
                        learning_rate: float,
                        save_path: str,
                        device: torch.device,
                        early_stopping_patience: int = 20,
                        amp: bool = True,
                        x_val: np.ndarray = None,
                        val_file_paths: list = None,
                        num_workers: int = 4,
                        pin_memory: bool = True,
                        persist_workers: bool = True,
                        prefetch_factor: int = 4,
                        resume_checkpoint_path: str = None,
                        resume_history: dict = None,
                        initial_best_metric: float = None,):
    """
    Train a Jukebox VQ-VAE model.

    Args:
        model (JukeboxVQVAE): The Jukebox VQ-VAE model to train.
        x_train (np.ndarray): Training spectrogram data.
        train_file_paths (list): List of file paths corresponding to x_train.
        min_max_values (dict): Dictionary of min/max values for denormalization.
        data_variance (float): Variance of the training data for loss scaling.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        save_path (str): Path to save the trained model.
        device (torch.device): Device to run the training on (CPU or GPU).
        early_stopping_patience (int, optional): Patience for early stopping. Defaults to 20.
        amp (bool, optional): Whether to use automatic mixed precision (AMP) during training. Defaults to True.
        x_val (np.ndarray, optional): Validation spectrogram data.
        val_file_paths (list, optional): List of file paths corresponding to x_val.
        num_workers (int, optional): Number of worker threads for data loading. Defaults to 4.
        pin_memory (bool, optional): Whether to pin memory for faster data transfer. Defaults to True.
        persist_workers (bool, optional): Whether to persist worker processes. Defaults to True.
        prefetch_factor (int, optional): Number of batches to prefetch. Defaults to 4.
    """
    model.to(device)
    
    # Setup Validation Data
    early_stopping = None
    val_dataloader = None
    val_dataset = None

    def _loader_kwargs():
        kwargs = {
            'num_workers': num_workers,
            'pin_memory': pin_memory,
        }
        if num_workers > 0:
            kwargs['persistent_workers'] = persist_workers
            if prefetch_factor is not None:
                kwargs['prefetch_factor'] = prefetch_factor
        return kwargs

    if x_val is not None:
        if isinstance(x_val, (np.ndarray, list)):
            if len(x_val) > 0:
                print(f"Training with {len(x_train)} samples and validating with {len(x_val)} samples.")
                val_dataset = SpectrogramDataset(x_val)
                val_dataloader = DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False,
                    **_loader_kwargs(),
                )
                early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
        else:
            # Assume x_val is a Dataset
            print(f"Training with {len(x_train)} samples and validating with {len(x_val)} samples.")
            val_dataset = x_val
            val_dataloader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                **_loader_kwargs(),
            )
            early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
    
    if val_dataloader is None:
        print(f"Using all {len(x_train)} samples for training (no validation set provided).")

    if isinstance(x_train, (np.ndarray, list)):
        dataset = SpectrogramDataset(x_train)
    else:
        dataset = x_train
        
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        **_loader_kwargs(),
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner for potential speedup
    scaler = GradScaler(enabled=amp and device.type == 'cuda')  # For mixed precision training

    start_epoch = 0
    if resume_checkpoint_path:
        checkpoint = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        if 'model_state' not in checkpoint:
            raise KeyError(f"Checkpoint at {resume_checkpoint_path} does not contain 'model_state'.")

        model.load_state_dict(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = int(checkpoint.get('epoch', -1)) + 1
        print(f"Resumed training from checkpoint: {resume_checkpoint_path}")
        print(f"Starting from epoch index {start_epoch} (human epoch {start_epoch + 1}).")

    print("Model will be saved to :", save_path)
    
    # Initialize Callbacks
    model_checkpoint = None
    loss_plotter = None
    sample_generator = None

    if save_path:
        model_checkpoint = ModelCheckpoint(
            save_path,
            model,
            optimizer,
            mode="min",
            initial_best_score=initial_best_metric,
        )
        loss_plotter = LossPlotter(save_path)
        if resume_history:
            loss_plotter.set_history(resume_history)
        
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
            sample_paths = val_file_paths[:4] if val_file_paths else None
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
            sample_paths = train_file_paths[:4] if train_file_paths else None

        sample_min_max = []
        for fp in sample_paths:
            mm = find_min_max_for_path(fp, min_max_values, spectrograms_dir)
            if mm is None:
                print(f"Warning: Could not find min/max for {fp}. Using default 0-1.")
                mm = {"min": 0.0, "max": 1.0}
            sample_min_max.append(mm)
            
        sample_generator = SampleGenerator(model, samples, sample_min_max, os.path.dirname(save_path), device)

    best_val_loss = float('inf')

    if start_epoch >= epochs:
        print(
            f"Checkpoint epoch ({start_epoch}) is already >= configured epochs ({epochs}). "
            "Nothing to train."
        )
        if loss_plotter:
            loss_plotter.plot()
            loss_plotter.save_history()
        return model

    for epoch in range(start_epoch, epochs):
        model.train()

        optimizer.zero_grad(set_to_none=True)
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_vq_losses = [0.0, 0.0, 0.0]  # [top, bottom, middle]
        total_samples = 0
        skipped_non_finite_batches = 0
        
        for i, batch in enumerate(progress_bar):
            batch = batch.to(device)

            if not torch.isfinite(batch).all():
                skipped_non_finite_batches += 1
                print(f"Warning: non-finite values in training batch at epoch {epoch+1}. Skipping batch.")
                continue

            with autocast(device_type=device.type, enabled=scaler.is_enabled()):
                reconstructions, total_vq_loss, vq_losses_details = model(batch)
                vq_loss, codebook_loss, commitment_loss = vq_losses_details[0]  # Jukebox VQ-VAE has only one quantizer

                recon_loss = F.mse_loss(reconstructions, batch) / (2 * data_variance)
                loss = recon_loss + total_vq_loss
                loss = loss / grad_accum_steps

            if not torch.isfinite(loss):
                skipped_non_finite_batches += 1
                print(f"Warning: non-finite training loss at epoch {epoch+1}. Skipping optimizer step for this batch.")
                # We do not reset gradients here since we might have successfully accumulated some
                continue

            scaler.scale(loss).backward()
            
            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(dataloader):
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_size_current = batch.size(0)
            # Reconstruct original loss for metrics logging
            actual_loss = loss.item() * grad_accum_steps
            epoch_loss += actual_loss * batch_size_current
            epoch_recon_loss += recon_loss.item() * batch_size_current
            epoch_vq_losses[0] += vq_loss.item() * batch_size_current
            epoch_vq_losses[1] += codebook_loss.item() * batch_size_current
            epoch_vq_losses[2] += commitment_loss.item() * batch_size_current
            total_samples += batch_size_current

            progress_bar.set_postfix(loss=epoch_loss / total_samples)

        if skipped_non_finite_batches > 0:
            print(f"Epoch {epoch+1}: skipped {skipped_non_finite_batches} non-finite training batches.")

        # Calculate average losses for the epoch
        if total_samples == 0:
            print(f"Epoch {epoch+1}: no valid training batches remained after filtering non-finite values. Stopping training.")
            break

        avg_epoch_loss = epoch_loss / total_samples
        avg_recon_loss = epoch_recon_loss / total_samples
        avg_vq_losses = [v / total_samples for v in epoch_vq_losses]

        # Update Loss Plotter
        epoch_metrics = {'total': avg_epoch_loss, 'reconstruction_loss': avg_recon_loss}
        epoch_metrics['vq_loss'] = avg_vq_losses[0]
        epoch_metrics['codebook_loss'] = avg_vq_losses[1]
        epoch_metrics['commitment_loss'] = avg_vq_losses[2]
        # Validation Loop
        val_loss_str = ""
        avg_val_loss = None
        
        if val_dataloader:
            model.eval()
            val_epoch_loss = 0.0
            val_epoch_recon_loss = 0.0
            val_epoch_vq_losses = [0.0, 0.0, 0.0]  # [vq_loss, codebook_loss, commitment]
            val_total_samples = 0
            skipped_non_finite_val_batches = 0
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    batch = batch.to(device)

                    if not torch.isfinite(batch).all():
                        skipped_non_finite_val_batches += 1
                        print(f"Warning: non-finite values in validation batch at epoch {epoch+1}. Skipping batch.")
                        continue

                    with autocast(device_type=device.type, enabled=scaler.is_enabled()):
                        reconstructions, total_vq_loss, vq_losses_details = model(batch)
                        vq_loss, codebook_loss, commitment_loss = vq_losses_details[0]  # Jukebox VQ-VAE has only one quantizer
                        recon_loss = F.mse_loss(reconstructions, batch) / (2 * data_variance)
                        loss = recon_loss + total_vq_loss
                    loss = recon_loss + total_vq_loss

                    if not torch.isfinite(loss):
                        skipped_non_finite_val_batches += 1
                        print(f"Warning: non-finite validation loss at epoch {epoch+1}. Skipping batch.")
                        continue
                    
                    batch_size_current = batch.size(0)
                    val_epoch_loss += loss.item() * batch_size_current
                    val_epoch_recon_loss += recon_loss.item() * batch_size_current
                    val_epoch_vq_losses[0] += vq_loss.item() * batch_size_current
                    val_epoch_vq_losses[1] += codebook_loss.item() * batch_size_current
                    val_epoch_vq_losses[2] += commitment_loss.item() * batch_size_current
                    val_total_samples += batch_size_current

            if skipped_non_finite_val_batches > 0:
                print(f"Epoch {epoch+1}: skipped {skipped_non_finite_val_batches} non-finite validation batches.")

            if val_total_samples == 0:
                avg_val_loss = float('nan')
                avg_val_vq_losses_val = [float('nan'), float('nan'), float('nan')]
            else:
                avg_val_loss = val_epoch_loss / val_total_samples
                avg_val_vq_losses_val = [v / val_total_samples for v in val_epoch_vq_losses]

            epoch_metrics['val_total'] = avg_val_loss
            epoch_metrics['val_reconstruction_loss'] = val_epoch_recon_loss / val_total_samples if val_total_samples else float('nan')
            epoch_metrics['val_vq_loss'] = avg_val_vq_losses_val[0]
            epoch_metrics['val_codebook_loss'] = avg_val_vq_losses_val[1]
            epoch_metrics['val_commitment_loss'] = avg_val_vq_losses_val[2]

            val_loss_str = f", Val Loss: {avg_val_loss:.4f}"

        vq_str = f"VQ Top: {avg_vq_losses[0]:.4f}, VQ Bottom: {avg_vq_losses[1]:.4f}"
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}, Recon: {avg_recon_loss:.4f}, {vq_str}{val_loss_str}")

        # Callbacks Step
        if loss_plotter:
            loss_plotter.update(epoch_metrics)
            loss_plotter.plot()
            loss_plotter.save_history()
            
        if model_checkpoint:
            # Use validation loss if available, else training loss
            metric_to_monitor = avg_val_loss if avg_val_loss is not None else avg_epoch_loss
            model_checkpoint.step(
                epoch,
                avg_epoch_loss,
                metric_value=metric_to_monitor,
                extra_state={
                    'metric_value': metric_to_monitor,
                    'history': loss_plotter.history if loss_plotter else {},
                },
            )
            
        if sample_generator:
            sample_generator.step(epoch)

        # Early Stopping Check
        if early_stopping and avg_val_loss is not None:
            if not np.isfinite(avg_val_loss):
                print("Validation loss is non-finite. Stopping training to prevent unstable checkpointing.")
                break
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return model
