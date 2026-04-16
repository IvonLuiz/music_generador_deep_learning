import librosa
import numpy as np
from typing import Any

import torch
from torch import nn as _nn

from processing.preprocess_audio import MinMaxNormalizer

class SoundGenerator:

    def __init__(self, autoencoder, hop_length, sample_rate=22050, n_fft=512, spectrogram_type="linear", n_mels=256):
        self.autoencoder = autoencoder 
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.spectrogram_type = str(spectrogram_type).strip().lower()
        if self.spectrogram_type not in ("linear", "mel"):
            raise ValueError(f"Unsupported spectrogram_type '{spectrogram_type}'. Expected 'linear' or 'mel'.")
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
                recon_out = model.reconstruct(x)
                if isinstance(recon_out, tuple):
                    x_hat = recon_out[0]
                    z_q = recon_out[1] if len(recon_out) > 1 else recon_out[0]
                else:
                    x_hat = recon_out
                    z_q = recon_out
                x_hat = x_hat.detach().cpu().permute(0, 2, 3, 1).numpy()  # (N,H,W,1)
                # For latent representations, return a detached cpu numpy copy
                latent_representations = z_q.detach().cpu().numpy()

            generated_spectrograms = x_hat
        else:
            # Fallback: assume the object implements reconstruct on numpy batches
            generated_spectrograms, latent_representations = self.autoencoder.reconstruct(spectrograms)

        signals = self.convert_spectrograms_to_audio(generated_spectrograms, min_max_values)

        return signals, latent_representations
    
    
    def convert_spectrograms_to_audio(self, spectrograms, min_max_values, method="griffinlim"):
        """
        Convert normalized log-spectrograms to audio signals.
        Args:
            spectrograms: Numpy array of shape (N, H, W, 1) with values in [0, 1]
            min_max_values: List of dicts with "min" and "max" for denormalization
        Returns:
            List of audio signals as numpy arrays
        """
        assert method in ("griffinlim", "istft"), "Unsupported inversion method. Use 'griffinlim' or 'istft'."

        signals = []

        for spectrogram, min_max_value in zip(spectrograms, min_max_values):
            # Reshape the log spectrogram to remove the third dimension (channels) used for the autoencoder
            log_spectrogram = spectrogram[:, :, 0]
            log_spectrogram = np.nan_to_num(log_spectrogram, nan=0.0, posinf=1.0, neginf=0.0)
            
            denorm_log_spec = self.__min_max_normalizer.denormalize(
                log_spectrogram, min_max_value["min"], min_max_value["max"]
            )
            denorm_log_spec = np.nan_to_num(denorm_log_spec, nan=0.0, posinf=80.0, neginf=-120.0)

            signal = self.__invert_log_spectrogram_to_audio(denorm_log_spec, method=method)
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
        if self.spectrogram_type == "mel":
            # Mel extractor uses power_to_db, so invert with db_to_power.
            mel_power = librosa.db_to_power(log_spectrogram)
            mel_power = np.nan_to_num(mel_power, nan=0.0, posinf=1e6, neginf=0.0)
            mel_power = np.maximum(mel_power, 0.0)

            if method == 'griffinlim':
                audio_signal = librosa.feature.inverse.mel_to_audio(
                    M=mel_power,
                    sr=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    power=2.0,
                )
            elif method == "istft":
                magnitude_stft = librosa.feature.inverse.mel_to_stft(
                    M=mel_power,
                    sr=self.sample_rate,
                    n_fft=self.n_fft,
                    power=2.0,
                )
                audio_signal = librosa.istft(magnitude_stft, hop_length=self.hop_length, n_fft=self.n_fft)
            else:
                raise ValueError(f"Unsupported inversion method: {method}")
        else:
            # Linear log-magnitude spectrogram path.
            amplitude_spectrogram = librosa.db_to_amplitude(log_spectrogram)
            amplitude_spectrogram = np.nan_to_num(amplitude_spectrogram, nan=0.0, posinf=1e6, neginf=0.0)
            amplitude_spectrogram = np.maximum(amplitude_spectrogram, 0.0)

            # If Nyquist bin was trimmed during preprocessing (256 bins for n_fft=512), restore it.
            if amplitude_spectrogram.shape[0] == self.n_fft // 2:
                amplitude_spectrogram = np.pad(amplitude_spectrogram, ((0, 1), (0, 0)), mode='constant')

            if method == 'griffinlim':
                audio_signal = librosa.griffinlim(amplitude_spectrogram, hop_length=self.hop_length, n_fft=self.n_fft)
            elif method == "istft":
                audio_signal = librosa.istft(amplitude_spectrogram, hop_length=self.hop_length, n_fft=self.n_fft)
            else:
                raise ValueError(f"Unsupported inversion method: {method}")

        audio_signal = np.nan_to_num(audio_signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        return audio_signal
