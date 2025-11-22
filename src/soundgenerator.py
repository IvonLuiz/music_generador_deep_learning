import librosa
import numpy as np
from typing import Any

import torch
from torch import nn as _nn

from processing.preprocess_audio import MinMaxNormalizer

class SoundGenerator:

    def __init__(self, autoencoder, hop_length):
        self.autoencoder = autoencoder 
        self.hop_length = hop_length
        self.__min_max_normalizer = MinMaxNormalizer(0, 1)

    
    def generate(self, spectrograms, min_max_values):
        """
        Generate audio signals from normalized log-spectrograms using the underlying autoencoder.

        Supports both PyTorch models (nn.Module) and previous TF-like objects that expose
        a reconstruct(...) API. Input spectrograms are expected as numpy arrays with
        shape (N, H, W, 1) and values in [0, 1].
        """

        # PyTorch path: convert numpy -> torch (N,1,H,W), run reconstruct, back to numpy (N,H,W,1)
        if _nn is not None and isinstance(self.autoencoder, _nn.Module):
            assert torch is not None, "Torch is required for PyTorch-based generation."
            # Prepare batch tensor
            if isinstance(spectrograms, np.ndarray):
                x = torch.from_numpy(spectrograms.astype(np.float32))  # (N, H, W, 1)
            else:
                # fallback if list of arrays
                x = torch.from_numpy(np.asarray(spectrograms, dtype=np.float32))
            x = x.permute(0, 3, 1, 2)  # (N,1,H,W)

            # Move to same device as the model
            device = next(self.autoencoder.parameters()).device
            self.autoencoder.eval()
            with torch.no_grad():
                x = x.to(device)
                model: Any = self.autoencoder
                x_hat, z_q = model.reconstruct(x)  # (N,1,H,W), (N, D, h, w)
                x_hat = x_hat.detach().cpu().permute(0, 2, 3, 1).numpy()  # (N,H,W,1)
                # For latent representations, return a detached cpu numpy copy
                latent_representations = z_q.detach().cpu().numpy()

            generated_spectrograms = x_hat
        else:
            # Fallback: assume the object implements reconstruct on numpy batches
            generated_spectrograms, latent_representations = self.autoencoder.reconstruct(spectrograms)

        signals = self.convert_spectrograms_to_audio(generated_spectrograms, min_max_values)

        return signals, latent_representations
    
    
    def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
        """
        Convert normalized log-spectrograms to audio signals.
        Args:
            spectrograms: Numpy array of shape (N, H, W, 1) with values in [0, 1]
            min_max_values: List of dicts with "min" and "max" for denormalization
        Returns:
            List of audio signals as numpy arrays
        """
        signals = []

        for spectrogram, min_max_value in zip(spectrograms, min_max_values):
            # Reshape the log spectrogram to remove the third dimension (channels) used for the autoencoder
            log_spectrogram = spectrogram[:, :, 0]
            
            denorm_log_spec = self.__min_max_normalizer.denormalize(
                log_spectrogram, min_max_value["min"], min_max_value["max"]
            )

            signal = self.__invert_log_spectrogram_to_audio(denorm_log_spec)
            signals.append(signal)

        return signals
    

    def __invert_log_spectrogram_to_audio(self, log_spectrogram, method="griffinlim"):
        """
        Invert a log-spectrogram back to audio using Griffin-Lim or ISTFT.
        Args:
            log_spectrogram: 2D numpy array (H, W) in dB scale
        Returns:
            1D numpy array audio signal
        """
        # Log spectrogram (dB) to amplitude domain
        amplitude_spectrogram = librosa.db_to_amplitude(log_spectrogram)
        
        # Pad the spectrogram to restore the Nyquist frequency bin if it was trimmed
        # STFT produces (n_fft // 2 + 1) bins, which is odd for even n_fft.
        # If we have an even number of bins (e.g. 256), it means the Nyquist bin was removed.
        if amplitude_spectrogram.shape[0] % 2 == 0:
             amplitude_spectrogram = np.pad(amplitude_spectrogram, ((0, 1), (0, 0)), mode='constant')

        if method =='griffinlim':
            # Use Griffin-Lim for phase estimation to improve audio quality
            audio_signal = librosa.griffinlim(amplitude_spectrogram, hop_length=self.hop_length)
        elif method == "istft":
            # STFT to get audio signal (robotic without phase reconstruction)
            audio_signal = librosa.istft(amplitude_spectrogram, hop_length=self.hop_length)
        
        return audio_signal
