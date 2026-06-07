from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from generation.audio_inversion import AudioGeometry, AudioInversionConfig
from generation.soundgenerator import SoundGenerator


class AudioExporter:
    """!
    @brief Converts normalized spectrogram batches to audio and saves waveforms.
    """

    def __init__(
        self,
        geometry: AudioGeometry,
        audio_config: Optional[AudioInversionConfig] = None,
        autoencoder=None,
    ):
        """!
        @brief Initialize an audio exporter.
        @param geometry STFT/Mel geometry.
        @param audio_config Spectrogram inversion settings.
        @param autoencoder Optional model for SoundGenerator compatibility.
        """
        self.geometry = geometry
        self.audio_config = audio_config or AudioInversionConfig(use_fixed_db_scale=True)
        self.sound_generator = SoundGenerator(
            autoencoder,
            hop_length=geometry.hop_length,
            sample_rate=geometry.sample_rate,
            n_fft=geometry.n_fft,
            spectrogram_type=geometry.spectrogram_type,
            n_mels=geometry.n_mels,
        )

    @staticmethod
    def fixed_min_max(count: int, min_db: float = -80.0, max_db: float = 0.0) -> List[Dict[str, float]]:
        """!
        @brief Build fixed dB denormalization metadata.
        @param count Number of spectrograms.
        @param min_db dB value represented by normalized 0.
        @param max_db dB value represented by normalized 1.
        @return List of min/max dictionaries.
        """
        return [{"min": float(min_db), "max": float(max_db)} for _ in range(int(count))]

    def convert(self, specs: np.ndarray, min_max_values: Optional[List[Dict[str, float]]] = None) -> List[np.ndarray]:
        """!
        @brief Convert normalized spectrograms into waveform arrays.
        @param specs Spectrogram batch shaped `(B, F, T, 1)`.
        @param min_max_values Optional per-sample denormalization metadata.
        @return List of waveform arrays.
        """
        if min_max_values is None or self.audio_config.use_fixed_db_scale:
            min_max_values = self.fixed_min_max(
                specs.shape[0],
                self.audio_config.fixed_min_db,
                self.audio_config.fixed_max_db,
            )
        return self.sound_generator.convert_spectrograms_to_audio(
            specs,
            min_max_values,
            inversion_config=self.audio_config,
        )

    def save_signals(self, signals_by_name: Dict[str, List[np.ndarray]], out_dir: str) -> List[str]:
        """!
        @brief Save grouped waveform arrays as `.wav` files.
        @param signals_by_name Mapping from group name to signal list.
        @param out_dir Output directory.
        @return List of written audio paths.
        """
        import soundfile as sf

        os.makedirs(out_dir, exist_ok=True)
        paths: List[str] = []
        for name, signals in signals_by_name.items():
            for idx, signal in enumerate(signals):
                path = os.path.join(out_dir, f"{name}_{idx:03d}.wav")
                sf.write(path, signal, self.geometry.sample_rate)
                paths.append(path)
        return paths
