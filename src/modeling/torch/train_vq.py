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
from soundgenerator import SoundGenerator
import soundfile as sf

from .vq_vae import VQ_VAE, vqvae_loss


class SpectrogramDataset(Dataset):
    def __init__(self, x: np.ndarray):
        # Expect (N, H, W, 1) with values in [0,1]
        assert x.ndim == 4 and x.shape[-1] == 1
        self.x = x.astype(np.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        spec = self.x[idx]  # (H, W, 1)
        # To torch (C,H,W)
        spec = np.transpose(spec, (2, 0, 1))  # (1, H, W)
        return torch.from_numpy(spec)


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
                amp: bool = True,
                grad_accum_steps: int = 1,
                max_grad_norm: Optional[float] = None,
                min_max_values: Optional[list] = None):

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
        amp=amp,
        grad_accum_steps=grad_accum_steps,
        max_grad_norm=max_grad_norm,
        model_config=config,
        min_max_values=min_max_values
    )


def train_model(model: VQ_VAE,
                x_train: np.ndarray,
                batch_size: int = 64,
                epochs: int = 50,
                learning_rate: float = 5e-4,
                data_variance: float = 1.0,
                save_path: Optional[str] = None,
                amp: bool = True,
                grad_accum_steps: int = 1,
                max_grad_norm: Optional[float] = None,
                model_config: Optional[dict] = None,
                min_max_values: Optional[list] = None):
    """
    Train an existing VQ-VAE model.
    
    Args:
        model: Pre-instantiated VQ_VAE model
        x_train: Training data of shape (N, H, W, 1)
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for Adam optimizer
        data_variance: Data variance for loss calculation
        save_path: Optional path to save the model after training
        model_config: Optional dictionary with model configuration to save
        min_max_values: Optional list of min/max values for reconstruction visualization
    
    Returns:
        The trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = model.to(device)
    ds = SpectrogramDataset(x_train)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    torch.backends.cudnn.benchmark = True
    scaler = GradScaler(enabled=(amp and device.type == 'cuda'))

    # Track losses for training progress
    train_losses_dict = {
        'total': [], 
        'codebook': [], 'commitment': [],
        'reconstruction': [], 'vq': []
    }
    
    print("Model will be saved to :", save_path) if save_path else None

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
                loss_full, recon_loss, vq_loss_val = vqvae_loss(specs, x_hat, vq_loss, variance=max(data_variance, 1e-6))
                loss = loss_full / grad_accum_steps
                
                # Accumulate individual losses for logging
                running_codebook_loss += codebook_loss.item() * specs.size(0)
                running_commitment_loss += commitment_loss.item() * specs.size(0)
                running_recon_loss += recon_loss.item() * specs.size(0)
                running_vq_loss += vq_loss_val.item() * specs.size(0)

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
        
        train_losses_dict['total'].append(running_loss / len(ds))
        train_losses_dict['codebook'].append(running_codebook_loss / len(ds))
        train_losses_dict['commitment'].append(running_commitment_loss / len(ds))
        train_losses_dict['reconstruction'].append(running_recon_loss / len(ds))
        train_losses_dict['vq'].append(running_vq_loss / len(ds))

        print(f"Epoch {epoch:03d}/{epochs} - losses: running {running_loss / len(ds):.6f}; codebook {running_codebook_loss / len(ds):.6f}, commitment {running_commitment_loss / len(ds):.6f}, recon {running_recon_loss / len(ds):.6f}, vq {running_vq_loss / len(ds):.6f}")

        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        if save_path:
            # Save model checkpoint
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_dict = {
                'model_state': model.state_dict(),
            }
            if model_config:
                save_dict['config'] = model_config
            
            torch.save(save_dict, save_path)
            plot_vqvae_losses(train_losses_dict, save_path=save_path)
            
            # Generate and save spectrograms for visualization
            epoch_save_dir = os.path.join(os.path.dirname(save_path), "samples", f"epoch_{epoch:03d}")
            # Take first 4 samples for visualization
            samples = x_train[:4]
            generate_and_save_spectrograms(model, samples, min_max_values, epoch_save_dir)

    return model


def generate_and_save_spectrograms(model: VQ_VAE,
                              specs: np.ndarray,
                              min_max_values: Optional[list],
                              save_dir: str):
    """Generate spectrograms from one of the audio samples before and after passing through the VQ-VAE and save."""
    sound_generator = SoundGenerator(model, hop_length=HOP_LENGTH)
    
    if min_max_values is None:
        # Create dummy min_max_values for visualization only (0-1 range)
        min_max_values = [{"min": 0.0, "max": 1.0} for _ in range(len(specs))]
        # print("Warning: min_max_values not provided. Visualizing normalized spectrograms (0-1).")

    os.makedirs(save_dir, exist_ok=True)
    save_spectrogram_comparisons(specs, min_max_values, sound_generator, save_dir=save_dir)


def save_spectrogram_comparisons(original_specs, min_max_values, sound_generator, save_dir="spectrograms/"):
    """
    Save side-by-side comparisons of original vs VQ-VAE reconstructed spectrograms.
    This visualizes how well the model reconstructs the input spectrograms.
    """
    from pathlib import Path
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Get VQ-VAE reconstructed spectrograms (in spectrogram domain, not audio)
    model = sound_generator.autoencoder
    model.eval()
    
    with torch.no_grad():
        # Convert to torch tensor and move to device
        if isinstance(original_specs, np.ndarray):
            x = torch.from_numpy(original_specs.astype(np.float32))
        else:
            x = torch.from_numpy(np.array(original_specs, dtype=np.float32))
        
        x = x.permute(0, 3, 1, 2)  # (N, 1, H, W)
        device = next(model.parameters()).device
        x = x.to(device)
        
        # Get reconstruction directly from VQ-VAE
        x_hat, z_q = model.reconstruct(x)
        reconstructed_specs = x_hat.cpu().permute(0, 2, 3, 1).numpy()  # Back to (N, H, W, 1)
    
    # Create comparison plots
    for i, (orig_spec, recon_spec, min_max_val) in enumerate(zip(original_specs, reconstructed_specs, min_max_values)):
        # Remove channel dimension for visualization
        if len(orig_spec.shape) == 3:
            orig_spec_2d = orig_spec[:, :, 0]
            recon_spec_2d = recon_spec[:, :, 0]
        else:
            orig_spec_2d = orig_spec
            recon_spec_2d = recon_spec
            
        # Denormalize both for proper visualization
        orig_min = min_max_val["min"]
        orig_max = min_max_val["max"]
        denorm_orig = orig_spec_2d * (orig_max - orig_min) + orig_min
        denorm_recon = recon_spec_2d * (orig_max - orig_min) + orig_min
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original spectrogram
        vmin, vmax = denorm_orig.min(), denorm_orig.max()
        im1 = axes[0].imshow(denorm_orig, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'Original Spectrogram\n(Sample {i+1})')
        axes[0].set_xlabel('Time Frames')
        axes[0].set_ylabel('Frequency Bins')
        plt.colorbar(im1, ax=axes[0], label='Magnitude (dB)')
        
        # VQ-VAE reconstructed spectrogram  
        im2 = axes[1].imshow(denorm_recon, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'VQ-VAE Reconstructed\n(Sample {i+1})')
        axes[1].set_xlabel('Time Frames')
        axes[1].set_ylabel('Frequency Bins')
        plt.colorbar(im2, ax=axes[1], label='Magnitude (dB)')
        
        # Handle shape mismatch by cropping to smaller dimension
        min_time_frames = min(denorm_orig.shape[1], denorm_recon.shape[1])
        denorm_orig_cropped = denorm_orig[:, :min_time_frames]
        denorm_recon_cropped = denorm_recon[:, :min_time_frames]
        
        # Difference (reconstruction error)
        diff = np.abs(denorm_orig_cropped - denorm_recon_cropped)
        im3 = axes[2].imshow(diff, aspect='auto', origin='lower', cmap='hot')
        axes[2].set_title(f'Reconstruction Error\n(Sample {i+1})\n(Cropped to {min_time_frames} frames)')
        axes[2].set_xlabel('Time Frames')
        axes[2].set_ylabel('Frequency Bins')
        plt.colorbar(im3, ax=axes[2], label='|Error| (dB)')
        
        # Add statistics as text
        mse = np.mean((denorm_orig_cropped - denorm_recon_cropped) ** 2)
        mae = np.mean(np.abs(denorm_orig_cropped - denorm_recon_cropped))
        
        # Add shape information to the title
        shape_info = f'Orig: {denorm_orig.shape}, Recon: {denorm_recon.shape}'
        fig.suptitle(f'MSE: {mse:.3f} dB², MAE: {mae:.3f} dB | {shape_info}', fontsize=10)
        
        plt.tight_layout()
        
        # Save the comparison
        save_path_img = os.path.join(save_dir, f"vqvae_comparison_{i+1:03d}.png")
        plt.savefig(save_path_img, dpi=150, bbox_inches='tight')
        plt.close()


def plot_vqvae_losses(train_losses_dict: dict, save_path: Optional[str] = None):

    os.makedirs(os.path.dirname(save_path), exist_ok=True) if save_path else None
    save_file_path = os.path.join(os.path.dirname(save_path), 'vqvae_losses.png') if save_path else None

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_dict['total'], label='Total Training Loss')
    plt.plot(train_losses_dict['vq'], label='VQ Loss')
    plt.plot(train_losses_dict['reconstruction'], label='Reconstruction Loss')
    plt.plot(train_losses_dict['codebook'], label='Codebook Loss')
    plt.plot(train_losses_dict['commitment'], label='Commitment Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VQ-VAE Loss Components over Epochs')
    plt.legend()
    if save_path:
        plt.savefig(save_file_path)
    else:
        plt.show()
    plt.close()

def plot_training_loss(loss_values, save_path: Optional[str] = None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if save_path else None
    save_file_path = os.path.join(os.path.dirname(save_path), 'training_loss.png') if save_path else None

    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    if save_path:
        plt.savefig(save_file_path)
    else:
        plt.show()
    plt.close()

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