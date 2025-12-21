import os
import pickle
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import matplotlib.pyplot as plt

from generation.soundgenerator import SoundGenerator
from modeling.torch.vq_vae import VQ_VAE
from train_scripts.train_vq_utils import load_fsdd


HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = "samples/original/"
SAVE_DIR_GENERATED = "samples/generated/"
MIN_MAX_VALUES_PATH = "./data/fsdd/min_max_values.pkl"


def select_spectrograms(spectrograms, file_paths, min_max_values, num_spectrograms=2):
    
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    print(sampled_indexes)
    sampled_spectrogrmas = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_min_max_values = [min_max_values[file_path] for file_path in file_paths]

    print(file_paths)
    print(sampled_min_max_values)
    
    return sampled_spectrogrmas, sampled_min_max_values


def save_signals(signals, save_dir, sample_rate=22050):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)

def save_multiple_signals(signals_dict, save_dir, sample_rate=22050):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for key, signals in signals_dict.items():
        key_dir = os.path.join(save_dir, key)
        Path(key_dir).mkdir(parents=True, exist_ok=True)
        save_signals(signals, key_dir, sample_rate)
        print('Saved', key, '->', key_dir)

def save_spectrogram_comparisons(original_specs, min_max_values, sound_generator, save_dir="spectrograms/"):
    """
    Save side-by-side comparisons of original vs VQ-VAE reconstructed spectrograms.
    This visualizes how well the model reconstructs the input spectrograms.
    """
    from pathlib import Path
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Get VQ-VAE reconstructed spectrograms (in spectrogram domain, not audio)
    with torch.no_grad():
        # Convert to torch tensor and move to device
        if isinstance(original_specs, np.ndarray):
            x = torch.from_numpy(original_specs.astype(np.float32))
        else:
            x = torch.from_numpy(np.array(original_specs, dtype=np.float32))
        
        x = x.permute(0, 3, 1, 2)  # (N, 1, H, W)
        device = next(sound_generator.autoencoder.parameters()).device
        x = x.to(device)
        
        # Get reconstruction directly from VQ-VAE
        x_hat, z_q = sound_generator.autoencoder.reconstruct(x)
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
        save_path = os.path.join(save_dir, f"vqvae_comparison_{i+1:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved VQ-VAE comparison: {save_path}")
        print(f"  MSE: {mse:.3f} dB², MAE: {mae:.3f} dB")

    print(f"\nAll spectrogram comparisons saved to: {save_dir}")


if __name__ == "__main__":
    vae = VQ_VAE.load("model")
    sound_generator = SoundGenerator(vae, HOP_LENGTH)

    # Load spectrograms + min max values
    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)
    specs, file_paths = load_fsdd(SPECTROGRAMS_PATH)

    # Sample spectrograms + min max values
    sampled_specs, sampled_min_max_values = select_spectrograms(specs, file_paths, min_max_values, 5)

    # Generate audio for sampled spectrograms
    signals, _ = sound_generator.generate(sampled_specs, sampled_min_max_values)

    original_signals = sound_generator.convert_spectrograms_to_audio(sampled_specs, sampled_min_max_values)

    save_signals(signals, SAVE_DIR_GENERATED)
    save_signals(original_signals, SAVE_DIR_ORIGINAL)