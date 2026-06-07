from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf


@dataclass
class EvaluationOutputConfig:
    """!
    @brief Output settings shared by evaluation scripts.
    """

    save_root: str
    run_name: str
    n_samples: int = 1
    seed: int = 42


@dataclass
class EvaluationResult:
    """!
    @brief Paths produced by an evaluation run.
    """

    output_dir: str
    audio_paths: List[str] = field(default_factory=list)
    spectrogram_paths: List[str] = field(default_factory=list)
    metadata_path: Optional[str] = None


class EvaluationRun:
    """!
    @brief Manages output directories and common artifact saving.
    """

    def __init__(self, config: EvaluationOutputConfig):
        """!
        @brief Create a timestamped evaluation run directory.
        @param config Output settings.
        """
        self.config = config
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = os.path.join(
            os.path.abspath(os.path.expanduser(config.save_root)),
            config.run_name,
            timestamp,
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.audio_paths: List[str] = []
        self.spectrogram_paths: List[str] = []
        self.metadata_path: Optional[str] = None

    def path(self, *parts: str) -> str:
        """!
        @brief Build a path under the run directory and create parent folders.
        @param parts Path components relative to the run directory.
        @return Absolute artifact path.
        """
        path = os.path.join(self.output_dir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def dir(self, *parts: str) -> str:
        """!
        @brief Build and create a directory under the run directory.
        @param parts Path components relative to the run directory.
        @return Absolute directory path.
        """
        path = os.path.join(self.output_dir, *parts)
        os.makedirs(path, exist_ok=True)
        return path

    def save_json(self, name: str, payload: Dict[str, Any]) -> str:
        """!
        @brief Save a JSON metadata artifact.
        @param name File name relative to the run directory.
        @param payload JSON-serializable payload.
        @return Written file path.
        """
        path = self.path(name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.metadata_path = path
        return path

    def save_array(self, name: str, array: np.ndarray) -> str:
        """!
        @brief Save a NumPy array artifact.
        @param name File name relative to the run directory.
        @param array Array to save.
        @return Written file path.
        """
        path = self.path(name)
        np.save(path, array)
        return path

    def save_audio(self, name: str, signal: np.ndarray, sample_rate: int) -> str:
        """!
        @brief Save one waveform artifact.
        @param name File name relative to the run directory.
        @param signal Waveform samples.
        @param sample_rate Audio sample rate in Hz.
        @return Written file path.
        """
        path = self.path(name)
        sf.write(path, signal, int(sample_rate))
        self.audio_paths.append(path)
        return path

    def metadata_payload(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """!
        @brief Build standard run metadata.
        @param extra Optional extra metadata fields.
        @return Metadata dictionary.
        """
        payload: Dict[str, Any] = {
            "output": asdict(self.config),
            "output_dir": self.output_dir,
        }
        if extra:
            payload.update(extra)
        return payload

    def result(self) -> EvaluationResult:
        """!
        @brief Return accumulated artifact paths.
        @return EvaluationResult for this run.
        """
        return EvaluationResult(
            output_dir=self.output_dir,
            audio_paths=list(self.audio_paths),
            spectrogram_paths=list(self.spectrogram_paths),
            metadata_path=self.metadata_path,
        )
