from typing import Any, Optional

import numpy as np
import torch
from torch import nn as _nn

from generation.audio_inversion import (
    SUPPORTED_AUDIO_METHODS,
    AudioGeometry,
    AudioInversionConfig,
    build_spectrogram_inverter,
)
from processing.preprocess_audio import MinMaxNormalizer


class SoundGenerator:
    """!
    @brief Reconstruct normalized spectrograms and convert them back to audio.

    The class intentionally keeps the old public API, but the inversion
    algorithms now live in generation.audio_inversion as separate classes.
    """

    def __init__(self, autoencoder, hop_length, sample_rate=22050, n_fft=512, spectrogram_type="linear", n_mels=256):
        """!
        @brief Create a reconstruction and spectrogram-inversion helper.
        @param autoencoder Model with `reconstruct` or None when only converting spectrograms.
        @param hop_length STFT/Mel hop length in samples.
        @param sample_rate Audio sample rate in Hz.
        @param n_fft FFT frame size.
        @param spectrogram_type Either `linear` or `mel`.
        @param n_mels Number of Mel bands when using Mel spectrograms.
        """
        self.autoencoder = autoencoder
        self.geometry = AudioGeometry(
            hop_length=hop_length,
            sample_rate=sample_rate,
            n_fft=n_fft,
            spectrogram_type=spectrogram_type,
            n_mels=n_mels,
        )
        self.hop_length = self.geometry.hop_length
        self.sample_rate = self.geometry.sample_rate
        self.n_fft = self.geometry.n_fft
        self.n_mels = self.geometry.n_mels
        self.spectrogram_type = self.geometry.spectrogram_type
        self.__min_max_normalizer = MinMaxNormalizer(0, 1)

    def generate(self, spectrograms, min_max_values, method="gradient", inversion_config=None, **inversion_kwargs):
        """!
        @brief Reconstruct normalized spectrograms with the autoencoder and invert them to audio.
        @param spectrograms Normalized spectrogram batch shaped `(N, H, W, 1)`.
        @param min_max_values List of denormalization dictionaries with `min` and `max`.
        @param method Legacy inversion method name used when inversion_config is absent.
        @param inversion_config Optional structured inversion settings.
        @param inversion_kwargs Legacy inversion keyword arguments.
        @return Tuple `(signals, latent_representations)`.
        """
        if _nn is not None and isinstance(self.autoencoder, _nn.Module):
            assert torch is not None, "Torch is required for PyTorch-based generation."
            x = (
                torch.from_numpy(spectrograms.astype(np.float32))
                if isinstance(spectrograms, np.ndarray)
                else torch.from_numpy(np.asarray(spectrograms, dtype=np.float32))
            )
            x = x.permute(0, 3, 1, 2)

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
                generated_spectrograms = x_hat.detach().cpu().permute(0, 2, 3, 1).numpy()
                latent_representations = z_q.detach().cpu().numpy()
        else:
            generated_spectrograms, latent_representations = self.autoencoder.reconstruct(spectrograms)

        signals = self.convert_spectrograms_to_audio(
            generated_spectrograms,
            min_max_values,
            method=method,
            inversion_config=inversion_config,
            **inversion_kwargs,
        )
        return signals, latent_representations

    def convert_spectrograms_to_audio(
        self,
        spectrograms,
        min_max_values,
        method="gradient",
        gradient_steps=1024,
        gradient_lr=0.0005,
        gradient_chunk_frames=8192,
        gradient_overlap_frames=2048,
        decorsiere_alpha=0.3,
        decorsiere_lr=1.0,
        decorsiere_history_size=10,
        inversion_config: Optional[AudioInversionConfig] = None,
    ):
        """!
        @brief Convert normalized log-spectrograms to audio signals.
        @param spectrograms Normalized spectrogram batch shaped `(N, H, W, 1)`.
        @param min_max_values List of denormalization dictionaries with `min` and `max`.
        @param method Legacy inversion method name used when inversion_config is absent.
        @param gradient_steps Legacy gradient/Decorsiere iteration count.
        @param gradient_lr Legacy Adam learning rate for gradient inversion.
        @param gradient_chunk_frames Legacy chunk length in spectrogram frames.
        @param gradient_overlap_frames Legacy chunk overlap in spectrogram frames.
        @param decorsiere_alpha Legacy compressed-envelope exponent.
        @param decorsiere_lr Legacy L-BFGS learning rate.
        @param decorsiere_history_size Legacy L-BFGS history size.
        @param inversion_config Optional structured inversion settings.
        @return List of recovered waveform arrays.
        """
        config = inversion_config or AudioInversionConfig(
            method=method,
            gradient_steps=gradient_steps,
            gradient_lr=gradient_lr,
            gradient_chunk_frames=gradient_chunk_frames,
            gradient_overlap_frames=gradient_overlap_frames,
            decorsiere_alpha=decorsiere_alpha,
            decorsiere_lr=decorsiere_lr,
            decorsiere_history_size=decorsiere_history_size,
        )
        inverter = build_spectrogram_inverter(config, self.geometry)
        signals = []

        for spectrogram, min_max_value in zip(spectrograms, min_max_values):
            log_spectrogram = spectrogram[:, :, 0]
            log_spectrogram = np.nan_to_num(log_spectrogram, nan=0.0, posinf=1.0, neginf=0.0)
            denorm_log_spec = self.__min_max_normalizer.denormalize(
                log_spectrogram,
                min_max_value["min"],
                min_max_value["max"],
            )
            denorm_log_spec = np.nan_to_num(denorm_log_spec, nan=0.0, posinf=80.0, neginf=-120.0)
            signals.append(inverter.invert(denorm_log_spec))

        return signals
