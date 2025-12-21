import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend to avoid thread issues
import matplotlib.pyplot as plt
import soundfile as sf

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import torch.nn.functional as F
from tqdm import tqdm

from modeling.torch.vq_vae_hierarchical import VQ_VAE_Hierarchical
from datasets.spectrogram_dataset import SpectrogramDataset
from generation.soundgenerator import SoundGenerator
from processing.preprocess_audio import HOP_LENGTH, SAMPLE_RATE
from utils import find_min_max_for_path


def train_vqvae_hierarquical(model: VQ_VAE_Hierarchical,
                             x_train: np.ndarray,
                             train_file_paths: list,
                             min_max_values: dict,
                             data_variance: float,
                             batch_size: int,
                             epochs: int,
                             learning_rate: float,
                             save_path: str,
                             device: torch.device,
                             amp: bool = True):
    """
    Train a VQ-VAE Hierarchical model.

    Args:
        model (VQ_VAE_Hierarchical): The VQ-VAE Hierarchical model to train.
        x_train (np.ndarray): Training spectrogram data.
        train_file_paths (list): List of file paths corresponding to x_train.
        min_max_values (dict): Dictionary of min/max values for denormalization.
        data_variance (float): Variance of the training data for loss scaling.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        model_file_path (str): Path to save the trained model.
        device (torch.device): Device to run the training on (CPU or GPU).
    """
    model.to(device)
    
    dataset = SpectrogramDataset(x_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner for potential speedup
    scaler = GradScaler(enabled=amp and device.type == 'cuda')  # For mixed precision training

    # Track losses for training progress
    train_losses_dict = {
        'total': [],
        'reconstruction_loss': [],
        'vq_loss_top': [],
        'vq_loss_bottom': [],
    }

    print("Model will be saved to :", save_path)
    
    # Create directory for saving samples
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad(set_to_none=True)
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

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

        train_losses_dict['total'].append(avg_epoch_loss)
        train_losses_dict['reconstruction_loss'].append(avg_recon_loss)
        train_losses_dict['vq_loss_top'].append(avg_vq_loss_top)
        train_losses_dict['vq_loss_bottom'].append(avg_vq_loss_bottom)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}, Recon: {avg_recon_loss:.4f}, VQ Top: {avg_vq_loss_top:.4f}, VQ Bottom: {avg_vq_loss_bottom:.4f}")

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        if save_path:
            # Save model checkpoint
            save_dict = {
                'model_state': model.state_dict(),
                'epoch': epoch,
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_epoch_loss
            }
            torch.save(save_dict, save_path)

            # Plot losses
            plot_vqvae_hierarchical_losses(train_losses_dict, save_path=save_path)

            # Generate and save spectrograms for visualization (every epoch)
            epoch_save_dir = os.path.join(os.path.dirname(save_path), "samples", f"epoch_{epoch+1:03d}")
            
            # Take first 4 samples for visualization
            samples = x_train[:4]
            sample_paths = train_file_paths[:4]
            
            # Find min/max values for these samples
            # We need the spectrograms directory to help find_min_max_for_path if needed, 
            # but usually the full path in sample_paths is enough.
            # We can infer spectrograms_dir from the first path if needed, or just pass empty string if paths are absolute.
            spectrograms_dir = os.path.dirname(sample_paths[0])
            
            sample_min_max = []
            for fp in sample_paths:
                mm = find_min_max_for_path(fp, min_max_values, spectrograms_dir)
                if mm is None:
                    print(f"Warning: Could not find min/max for {fp}. Using default 0-1.")
                    mm = {"min": 0.0, "max": 1.0}
                sample_min_max.append(mm)

            # We need to create a temporary wrapper or adapt the function since SoundGenerator might expect standard VQ-VAE
            # For now, let's manually call reconstruction
            generate_and_save_hierarchical_spectrograms(model, samples, sample_min_max, epoch_save_dir, device)

    return model

def plot_vqvae_hierarchical_losses(train_losses_dict: dict, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_file_path = os.path.join(os.path.dirname(save_path), 'vqvae_hierarchical_losses.png')

    plt.figure(figsize=(12, 6))
    plt.plot(train_losses_dict['total'], label='Total Loss')
    plt.plot(train_losses_dict['reconstruction_loss'], label='Reconstruction Loss')
    plt.plot(train_losses_dict['vq_loss_top'], label='VQ Loss Top')
    plt.plot(train_losses_dict['vq_loss_bottom'], label='VQ Loss Bottom')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VQ-VAE Hierarchical Loss Components')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_file_path)
    plt.close()

def generate_and_save_hierarchical_spectrograms(model, specs, min_max_values, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize SoundGenerator for audio conversion
    sound_generator = SoundGenerator(model, hop_length=HOP_LENGTH)

    model.eval()
    with torch.no_grad():
        # Prepare input
        if isinstance(specs, np.ndarray):
            x = torch.from_numpy(specs.astype(np.float32))
        else:
            x = torch.from_numpy(np.array(specs, dtype=np.float32))
        
        x = x.permute(0, 3, 1, 2).to(device) # (N, 1, H, W)
        
        # Reconstruct
        x_recon = model.reconstruct(x)
        reconstructed_specs = x_recon.cpu().permute(0, 2, 3, 1).numpy()
        
    # Convert to audio
    # Note: SoundGenerator expects (N, H, W, 1)
    # We use the original specs (normalized) and reconstructed specs (normalized)
    # SoundGenerator handles denormalization internally using min_max_values
    
    # Convert original spectrograms to audio
    original_signals = sound_generator.convert_spectrograms_to_audio(specs, min_max_values)
    
    # Convert reconstructed spectrograms to audio
    reconstructed_signals = sound_generator.convert_spectrograms_to_audio(reconstructed_specs, min_max_values)

    for i, (orig, recon, orig_sig, recon_sig) in enumerate(zip(specs, reconstructed_specs, original_signals, reconstructed_signals)):
        orig_2d = orig[:, :, 0]
        recon_2d = recon[:, :, 0]
        
        # Plotting normalized spectrograms (0-1 scale)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original
        im1 = axes[0].imshow(orig_2d, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1)
        axes[0].set_title(f'Original {i} (Normalized)')
        plt.colorbar(im1, ax=axes[0])
        
        # Recon
        im2 = axes[1].imshow(recon_2d, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title(f'Reconstructed {i} (Normalized)')
        plt.colorbar(im2, ax=axes[1])
        
        # Diff
        diff = np.abs(orig_2d - recon_2d)
        im3 = axes[2].imshow(diff, origin='lower', aspect='auto', cmap='hot', vmin=0, vmax=0.4)
        axes[2].set_title(f'Difference {i}')
        plt.colorbar(im3, ax=axes[2])

        # Add statistics as text
        mse = np.mean((orig_2d - recon_2d) ** 2)
        mae = np.mean(np.abs(orig_2d - recon_2d))
        
        # Add shape information to the title
        shape_info = f'Orig: {orig_2d.shape}, Recon: {recon_2d.shape}'
        fig.suptitle(f'MSE: {mse:.6f}, MAE: {mae:.6f} | {shape_info}', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"comparison_{i}.png"))
        plt.close()

        # Save audio
        sf.write(os.path.join(save_dir, f"original_{i}.wav"), orig_sig, SAMPLE_RATE)
        sf.write(os.path.join(save_dir, f"reconstructed_{i}.wav"), recon_sig, SAMPLE_RATE)
