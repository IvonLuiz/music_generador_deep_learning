from typing import Optional

import numpy as np

from generation.audio_inversion import (
    SUPPORTED_AUDIO_METHODS,
    AudioGeometry,
    AudioInversionConfig,
    build_spectrogram_inverter,
)
from processing.preprocess_audio import MinMaxNormalizer


class SpectrogramAudioConverter:
    """!
    @brief Convert normalized spectrograms back to audio waveforms.
    """

    def __init__(self, hop_length, sample_rate=22050, n_fft=512, spectrogram_type="linear", n_mels=256):
        """!
        @brief Create a spectrogram inversion helper.
        @param hop_length STFT/Mel hop length in samples.
        @param sample_rate Audio sample rate in Hz.
        @param n_fft FFT frame size.
        @param spectrogram_type Either `linear` or `mel`.
        @param n_mels Number of Mel bands when using Mel spectrograms.
        """
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
