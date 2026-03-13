import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from generation.soundgenerator import SoundGenerator
from processing.preprocess_audio import HOP_LENGTH, SAMPLE_RATE

class SampleGenerator:
    """
    Generates and saves spectrograms and audio samples during training.
    """
    def __init__(self, model, samples, min_max_values, save_dir, device):
        """
        Args:
            model: The VQ-VAE model.
            samples (np.ndarray): A batch of samples to reconstruct.
            min_max_values (list): List of min/max dictionaries for each sample.
            save_dir (str): Directory to save samples.
            device (torch.device): Device to run inference on.
        """
        self.model = model
        self.samples = samples
        self.min_max_values = min_max_values
        self.save_dir = save_dir
        self.device = device
        self.sound_generator = SoundGenerator(model, hop_length=HOP_LENGTH)

    def step(self, epoch):
        epoch_save_dir = os.path.join(self.save_dir, "samples", f"epoch_{epoch+1:03d}")
        os.makedirs(epoch_save_dir, exist_ok=True)
        
        self.model.eval()
        with torch.no_grad():
            # Prepare input
            if isinstance(self.samples, np.ndarray):
                x = torch.from_numpy(self.samples.astype(np.float32))
            else:
                x = torch.from_numpy(np.array(self.samples, dtype=np.float32))
            
            x = x.permute(0, 3, 1, 2).to(self.device) # (N, 1, H, W)
            
            # Reconstruct
            reconstruction_output = self.model.reconstruct(x)
            
            # Handle different return types from reconstruct
            if isinstance(reconstruction_output, tuple):
                x_recon = reconstruction_output[0]
            else:
                x_recon = reconstruction_output
                
            reconstructed_specs = x_recon.cpu().permute(0, 2, 3, 1).numpy()

        # Keep spectrograms finite before Griffin-Lim inversion
        safe_original_specs = np.nan_to_num(self.samples, nan=0.0, posinf=1.0, neginf=0.0)
        safe_original_specs = np.clip(safe_original_specs, 0.0, 1.0)
        reconstructed_specs = np.nan_to_num(reconstructed_specs, nan=0.0, posinf=1.0, neginf=0.0)
        reconstructed_specs = np.clip(reconstructed_specs, 0.0, 1.0)
            
        # Convert to audio
        try:
            original_signals = self.sound_generator.convert_spectrograms_to_audio(safe_original_specs, self.min_max_values)
        except Exception as e:
            print(f"Warning: failed to convert original spectrograms to audio at epoch {epoch+1}: {e}")
            original_signals = [np.zeros(SAMPLE_RATE, dtype=np.float32) for _ in range(len(safe_original_specs))]

        try:
            reconstructed_signals = self.sound_generator.convert_spectrograms_to_audio(reconstructed_specs, self.min_max_values)
        except Exception as e:
            print(f"Warning: failed to convert reconstructed spectrograms to audio at epoch {epoch+1}: {e}")
            reconstructed_signals = [np.zeros(SAMPLE_RATE, dtype=np.float32) for _ in range(len(reconstructed_specs))]

        for i, (orig, recon, orig_sig, recon_sig, min_max_val) in enumerate(zip(safe_original_specs, reconstructed_specs, original_signals, reconstructed_signals, self.min_max_values)):
            orig_2d = orig[:, :, 0]
            recon_2d = recon[:, :, 0]
            
            # Create comparison plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Original
            im1 = axes[0].imshow(orig_2d, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1)
            axes[0].set_title(f'Original Spectrogram\n(Sample {i+1})')
            axes[0].set_xlabel('Time Frames')
            axes[0].set_ylabel('Frequency Bins')
            plt.colorbar(im1, ax=axes[0], label='Normalized Magnitude')
            
            # Recon
            im2 = axes[1].imshow(recon_2d, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1)
            axes[1].set_title(f'VQ-VAE Reconstructed\n(Sample {i+1})')
            axes[1].set_xlabel('Time Frames')
            axes[1].set_ylabel('Frequency Bins')
            plt.colorbar(im2, ax=axes[1], label='Normalized Magnitude')
            
            # Handle shape mismatch
            min_time_frames = min(orig_2d.shape[1], recon_2d.shape[1])
            orig_cropped = orig_2d[:, :min_time_frames]
            recon_cropped = recon_2d[:, :min_time_frames]

            # Diff
            diff = np.abs(orig_cropped - recon_cropped)
            im3 = axes[2].imshow(diff, origin='lower', aspect='auto', cmap='hot', vmin=0, vmax=0.4)
            axes[2].set_title(f'Reconstruction Error\n(Sample {i+1})\n(Cropped to {min_time_frames} frames)')
            axes[2].set_xlabel('Time Frames')
            axes[2].set_ylabel('Frequency Bins')
            plt.colorbar(im3, ax=axes[2], label='|Error|')

            # Add statistics as text
            mse = np.mean((orig_cropped - recon_cropped) ** 2)
            mae = np.mean(np.abs(orig_cropped - recon_cropped))
            
            # Add shape information to the title
            shape_info = f'Orig: {orig_2d.shape}, Recon: {recon_2d.shape}'
            fig.suptitle(f'MSE: {mse:.6f}, MAE: {mae:.6f} | {shape_info}', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_save_dir, f"comparison_{i+1:03d}.png"), dpi=150, bbox_inches='tight')
            plt.close()

            # Save audio
            sf.write(os.path.join(epoch_save_dir, f"original_{i+1:03d}.wav"), orig_sig, SAMPLE_RATE)
            sf.write(os.path.join(epoch_save_dir, f"reconstructed_{i+1:03d}.wav"), recon_sig, SAMPLE_RATE)
