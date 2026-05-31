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

from processing.preprocess_audio import HOP_LENGTH, SAMPLE_RATE, FRAME_SIZE, N_MELS
from generation.soundgenerator import SoundGenerator
import soundfile as sf

from modeling.torch.vq_vae import VQ_VAE, vqvae_loss
from datasets.spectrogram_dataset import SpectrogramDataset
from callbacks import EarlyStopping, ModelCheckpoint, LossPlotter, SampleGenerator
from train_scripts.train_vqvae_utils import (
    _collect_preprocessed_callback_samples,
    _estimate_preprocessed_variance,
    _move_batch_to_device,
)


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
                data_variance: Optional[float] = 1.0,
                early_stopping_patience: int = 20,
                save_path: Optional[str] = None,
                amp: bool = True,
                grad_accum_steps: int = 1,
                max_grad_norm: Optional[float] = None,
                model_config: Optional[dict] = None,
                min_max_values: Optional[list] = None,
                x_val: Optional[np.ndarray] = None,
                num_workers: int = 4,
                pin_memory: bool = True,
                persist_workers: bool = True,
                prefetch_factor: Optional[int] = 4,
                spectrogram_type: str = 'linear',
                sample_rate: int = SAMPLE_RATE,
                hop_length: int = HOP_LENGTH,
                frame_size: int = FRAME_SIZE,
                n_mels: int = N_MELS,
                batch_preprocessor: Optional[torch.nn.Module] = None,
                collate_fn=None,
                data_variance_samples: int = 1000,
                resume_checkpoint_path: Optional[str] = None,
                resume_history: Optional[dict] = None,
                initial_best_metric: Optional[float] = None):
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
        batch_preprocessor: Optional module that converts raw dataloader batches
            into model-ready spectrogram tensors on the training device.
        collate_fn: Optional dataloader collate function, required by raw-audio
            datasets that return dictionaries.
    
    Returns:
        The trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = model.to(device)
    if batch_preprocessor is not None:
        batch_preprocessor.to(device)

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

    def _prepare_batch(batch, augment: bool):
        batch = _move_batch_to_device(batch, device)
        if batch_preprocessor is not None:
            batch = batch_preprocessor(batch, augment=augment)
        return batch
    
    if isinstance(x_train, (np.ndarray, list)):
        ds = SpectrogramDataset(x_train)
    else:
        ds = x_train
        
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **_loader_kwargs(),
    )

    # Setup Validation Data
    early_stopping = None
    val_dataloader = None
    val_dataset = None

    historical_counter = 0
    if resume_history and initial_best_metric is not None:
        val_losses_history = resume_history.get('val_total', resume_history.get('val_loss', []))
        for loss in reversed(val_losses_history):
            if loss > initial_best_metric:
                historical_counter += 1
            else:
                break
    initial_best_score = -initial_best_metric if initial_best_metric is not None else None

    if x_val is not None:
        if isinstance(x_val, (np.ndarray, list)):
            if len(x_val) > 0:
                print(f"Training with {len(x_train)} samples and validating with {len(x_val)} samples.")
                val_dataset = SpectrogramDataset(x_val)
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                    **_loader_kwargs(),
                )
                early_stopping = EarlyStopping(
                    patience=early_stopping_patience,
                    verbose=True,
                    best_score=initial_best_score,
                    counter=historical_counter,
                )
        else:
            # Assume x_val is a Dataset
            print(f"Training with {len(x_train)} samples and validating with {len(x_val)} samples.")
            val_dataset = x_val
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                **_loader_kwargs(),
            )
            early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                verbose=True,
                best_score=initial_best_score,
                counter=historical_counter,
            )
    
    if val_dataloader is None:
        print(f"Using all {len(x_train)} samples for training (no validation set provided).")

    if data_variance is None:
        if batch_preprocessor is None:
            raise ValueError("data_variance can only be None when a batch_preprocessor is provided.")
        data_variance = _estimate_preprocessed_variance(
            dl,
            batch_preprocessor,
            device,
            max_samples=int(data_variance_samples),
        )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    torch.backends.cudnn.benchmark = True
    scaler = GradScaler(enabled=(amp and device.type == 'cuda'))
    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")
    print(f"Using batch size {batch_size} with gradient accumulation over {grad_accum_steps} steps for effective batch size of {batch_size * grad_accum_steps}.")

    start_epoch = 1
    if resume_checkpoint_path:
        checkpoint = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        if 'model_state' not in checkpoint:
            raise KeyError(f"Checkpoint at {resume_checkpoint_path} does not contain 'model_state'.")

        model.load_state_dict(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])

        if data_variance is None and 'data_variance' in checkpoint:
            data_variance = float(checkpoint['data_variance'])

        if 'epoch' in checkpoint:
            completed_epoch = int(checkpoint['epoch'])
            start_epoch = completed_epoch + 1
            print(f"Resumed training from checkpoint: {resume_checkpoint_path}")
            print(f"Checkpoint completed epoch {completed_epoch}. Starting from epoch {start_epoch}.")
        else:
            completed_epochs = len(resume_history.get('val_total', resume_history.get('total', [])) if resume_history else [])
            start_epoch = completed_epochs + 1
            print(f"Checkpoint did not contain an epoch key. Inferred start_epoch {start_epoch} from history.")

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
        sample_source = val_dataset if val_dataloader else ds
        if batch_preprocessor is not None:
            samples, sample_min_max = _collect_preprocessed_callback_samples(
                sample_source,
                batch_preprocessor,
                collate_fn,
                device,
                sample_count=4,
            )
        elif val_dataloader:
            if isinstance(val_dataset, Dataset):
                samples = []
                for i in range(min(4, len(val_dataset))):
                    s = val_dataset[i]
                    s = s.permute(1, 2, 0).numpy()
                    samples.append(s)
                samples = np.stack(samples)
            else:
                samples = x_val[:4]
            sample_min_max = min_max_values[:len(samples)] if min_max_values is not None else None
        else:
            if isinstance(ds, Dataset):
                samples = []
                for i in range(min(4, len(ds))):
                    s = ds[i]
                    s = s.permute(1, 2, 0).numpy()
                    samples.append(s)
                samples = np.stack(samples)
            else:
                samples = x_train[:4]
            sample_min_max = min_max_values[:len(samples)] if min_max_values is not None else None
        
        if sample_min_max is None:
            # Create dummy min_max_values for visualization only (0-1 range)
            sample_min_max = [{"min": 0.0, "max": 1.0} for _ in range(len(samples))]
        
        sample_generator = SampleGenerator(
            model,
            samples,
            sample_min_max,
            os.path.dirname(save_path),
            device,
            spectrogram_type=spectrogram_type,
            hop_length=hop_length,
            sample_rate=sample_rate,
            n_fft=frame_size,
            n_mels=n_mels,
        )

    if start_epoch > epochs or (early_stopping is not None and early_stopping.counter >= early_stopping.patience):
        print(
            f"Checkpoint epoch ({start_epoch - 1}) is already >= configured epochs ({epochs}) "
            f"or patience ({early_stopping.patience if early_stopping else 'N/A'}) has been exhausted "
            f"(counter: {early_stopping.counter if early_stopping else 'N/A'}). Nothing to train."
        )
        if loss_plotter:
            loss_plotter.plot()
            loss_plotter.save_history()
        return model

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss, running_codebook_loss, running_commitment_loss, running_recon_loss, running_vq_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        total_samples = 0
        optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(dl, desc=f"Epoch {epoch:03d}/{epochs}")
        remainder_batches = len(dl) % grad_accum_steps
        first_remainder_step = len(dl) - remainder_batches + 1 if remainder_batches else None
        for step, specs in enumerate(progress_bar, start=1):
            specs = _prepare_batch(specs, augment=True)
            current_accum_steps = (
                remainder_batches
                if first_remainder_step is not None and step >= first_remainder_step
                else grad_accum_steps
            )
            with autocast(device_type=device.type, enabled=scaler.is_enabled()):
                x_hat, _z, vq_loss, codebook_loss, commitment_loss = model(specs)
                loss_full, recon_loss  = vqvae_loss(specs, x_hat, vq_loss, variance=max(data_variance, 1e-6))
                loss = loss_full / current_accum_steps
                
                # Accumulate individual losses for logging
                running_codebook_loss += codebook_loss.item() * specs.size(0)
                running_commitment_loss += commitment_loss.item() * specs.size(0)
                running_recon_loss += recon_loss.item() * specs.size(0)
                running_vq_loss += vq_loss.item() * specs.size(0)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if step % grad_accum_steps == 0 or step == len(dl):
                    if max_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if step % grad_accum_steps == 0 or step == len(dl):
                    if max_grad_norm is not None:
                        clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            batch_size_current = specs.size(0)
            running_loss += loss_full.item() * batch_size_current
            total_samples += batch_size_current
            progress_bar.set_postfix(loss=running_loss / total_samples)
        
        avg_loss = running_loss / total_samples
        avg_codebook = running_codebook_loss / total_samples
        avg_commitment = running_commitment_loss / total_samples
        avg_recon = running_recon_loss / total_samples
        avg_vq = running_vq_loss / total_samples

        # Validation Loop
        val_loss_str = ""
        avg_val_loss = None
        
        if val_dataloader:
            model.eval()
            val_running_loss = 0.0
            val_running_recon_loss = 0.0
            val_running_vq_loss = 0.0
            val_total_samples = 0
            
            val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch:03d}/{epochs} [Val]")
            with torch.no_grad():
                for val_specs in val_progress_bar:
                    val_specs = _prepare_batch(val_specs, augment=False)
                    x_hat, _z, vq_loss, codebook_loss, commitment_loss = model(val_specs)
                    loss_full, recon_loss = vqvae_loss(val_specs, x_hat, vq_loss, variance=max(data_variance, 1e-6))
                    
                    batch_size_current = val_specs.size(0)
                    val_running_loss += loss_full.item() * batch_size_current
                    val_running_recon_loss += recon_loss.item() * batch_size_current
                    val_running_vq_loss += vq_loss.item() * batch_size_current
                    val_total_samples += batch_size_current

                    val_progress_bar.set_postfix(
                        val_loss=val_running_loss / val_total_samples,
                        val_recon=val_running_recon_loss / val_total_samples,
                    )
            
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
            loss_plotter.save_history()
            
        if model_checkpoint:
            metric_to_monitor = avg_val_loss if avg_val_loss is not None else avg_loss
            model_checkpoint.step(
                epoch,
                avg_loss,
                metric_value=metric_to_monitor,
                extra_state={
                    'metric_value': metric_to_monitor,
                    'history': loss_plotter.history if loss_plotter else {},
                    'data_variance': data_variance,
                },
            )
            
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
