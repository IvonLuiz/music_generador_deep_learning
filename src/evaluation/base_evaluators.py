from __future__ import annotations

import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from evaluation.audio import AudioExporter
from evaluation.core import EvaluationOutputConfig, EvaluationResult, EvaluationRun
from evaluation.visualization import SpectrogramPlotConfig, SpectrogramVisualizer
from generation.audio_inversion import (
    DEFAULT_FIXED_MAX_DB,
    DEFAULT_FIXED_MIN_DB,
    AudioGeometry,
    AudioInversionConfig,
)
from processing.preprocess_audio import FRAME_SIZE, HOP_LENGTH, N_MELS, SAMPLE_RATE
from utils import find_min_max_for_path


@dataclass
class ReconstructionPayload:
    """!
    @brief Data produced by a reconstruction evaluator before common artifact export.
    """

    run_config: dict
    original_specs: np.ndarray
    reconstructed_specs: np.ndarray
    min_max_values: Optional[List[dict]]
    sampled_paths: List[str]
    metadata: Dict = field(default_factory=dict)
    codebook_indices: Optional[np.ndarray] = None
    codebook_num_embeddings: Optional[int] = None
    save_sampled_paths_array: bool = False
    include_sampled_paths_metadata: bool = True


@dataclass
class GenerationPayload:
    """!
    @brief Data produced by a prior evaluator before common artifact export.
    """

    run_config: dict
    specs: np.ndarray
    min_max_values: Optional[List[dict]] = None
    indices: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    spectrogram_prefix: str = "generated_specs"
    spectrogram_title: str = "Generated spectrogram"


class BaseEvaluator(ABC):
    """!
    @brief Shared setup and helper methods for evaluation classes.
    """

    run_name: str = "evaluation"

    def __init__(self, config, audio_config: Optional[AudioInversionConfig] = None):
        """!
        @brief Store evaluator configuration and prepare audio inversion settings.
        @param config Evaluation dataclass.
        @param audio_config Optional explicit audio inversion config.
        """
        self.config = config
        self.audio_config = audio_config or self._default_audio_config()

    def _default_audio_config(self) -> AudioInversionConfig:
        """!
        @brief Build default audio inversion settings from common config fields.
        @return AudioInversionConfig instance.
        """
        return AudioInversionConfig(
            method=getattr(self.config, "audio_method", "gradient"),
            use_fixed_db_scale=True,
            fixed_min_db=getattr(self.config, "min_db", DEFAULT_FIXED_MIN_DB),
            fixed_max_db=getattr(self.config, "max_db", DEFAULT_FIXED_MAX_DB),
        )

    @staticmethod
    def _device() -> torch.device:
        """!
        @brief Return the evaluation device.
        @return CUDA device when available, otherwise CPU.
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _config_path_for(model_path: str) -> str:
        """!
        @brief Resolve config.yaml next to a model reference.
        @param model_path Run directory, config file, or checkpoint file.
        @return Config path.
        """
        if os.path.isdir(model_path):
            return os.path.join(model_path, "config.yaml")
        if os.path.basename(model_path).lower() in ("config.yaml", "config.yml"):
            return model_path
        return os.path.join(os.path.dirname(model_path), "config.yaml")

    @staticmethod
    def _audio_geometry_from_config(config: dict) -> AudioGeometry:
        """!
        @brief Build audio geometry from a model config.
        @param config Parsed model config.
        @return AudioGeometry instance.
        """
        dataset_cfg = config.get("dataset", {})
        processed_path = dataset_cfg.get("processed_path", "")
        spectrogram_type_cfg = dataset_cfg.get("spectrogram_type")
        spectrogram_type = str(spectrogram_type_cfg).strip().lower() if spectrogram_type_cfg else (
            "mel" if "mel" in str(processed_path).lower() else "linear"
        )
        return AudioGeometry(
            hop_length=int(dataset_cfg.get("hop_length", HOP_LENGTH)),
            sample_rate=int(dataset_cfg.get("sample_rate", SAMPLE_RATE)),
            n_fft=int(dataset_cfg.get("frame_size", FRAME_SIZE)),
            spectrogram_type=spectrogram_type,
            n_mels=int(dataset_cfg.get("n_mels", N_MELS)),
        )

    @staticmethod
    def _min_max_for_paths(
        sampled_paths: List[str],
        spectrograms_path: str,
        min_max_values_path: Optional[str],
        fallback_min_db: float,
        fallback_max_db: float,
    ) -> Optional[List[dict]]:
        """!
        @brief Load explicit min/max metadata for sampled spectrogram paths.
        @return Per-sample min/max dictionaries when a file is provided, otherwise None.
        """
        if not min_max_values_path:
            return None
        with open(os.path.abspath(os.path.expanduser(min_max_values_path)), "rb") as f:
            min_max_values = pickle.load(f)
        result: List[dict] = []
        fallback = {"min": float(fallback_min_db), "max": float(fallback_max_db)}
        for path in sampled_paths:
            result.append(find_min_max_for_path(path, min_max_values, spectrograms_path) or fallback)
        return result

    def _create_run(self, n_samples: int, run_name: Optional[str] = None, seed: Optional[int] = None) -> EvaluationRun:
        """!
        @brief Create a timestamped output run.
        @param n_samples Number of evaluated/generated samples.
        @param run_name Optional run name override.
        @param seed Optional seed override.
        @return EvaluationRun instance.
        """
        return EvaluationRun(
            EvaluationOutputConfig(
                self.config.save_root,
                run_name or self.run_name,
                int(n_samples),
                42 if seed is None else seed,
            )
        )

    def _exporter(self, run_config: dict) -> AudioExporter:
        """!
        @brief Build an audio exporter for a model config.
        @param run_config Parsed model config.
        @return AudioExporter instance.
        """
        return AudioExporter(self._audio_geometry_from_config(run_config), self.audio_config)

    def _config_dict(self) -> dict:
        """!
        @brief Return the evaluator config as a dictionary.
        @return Dict form of the config dataclass or object.
        """
        try:
            return asdict(self.config)
        except TypeError:
            return dict(getattr(self.config, "__dict__", {}))


class VQVAEEvaluator(BaseEvaluator):
    """!
    @brief Shared run flow for reconstruction evaluators.
    """

    @abstractmethod
    def _evaluate_reconstruction(self, device: torch.device) -> ReconstructionPayload:
        """!
        @brief Produce reconstructed spectrograms and model-specific metadata.
        """

    def _reconstruction_audio_groups(self, original_signals, reconstructed_signals) -> Dict[str, List[np.ndarray]]:
        """!
        @brief Return named audio groups in the default reconstruction save order.
        """
        return {"reconstructed": reconstructed_signals, "original": original_signals}

    def _run_name_for_payload(self, payload: ReconstructionPayload) -> str:
        """!
        @brief Return the output run name for a reconstruction payload.
        """
        return self.run_name

    def run(self) -> EvaluationResult:
        """!
        @brief Run reconstruction evaluation and save common artifacts.
        @return Evaluation artifact paths.
        """
        payload = self._evaluate_reconstruction(self._device())
        exporter = self._exporter(payload.run_config)
        reconstructed_audio = exporter.convert(payload.reconstructed_specs, payload.min_max_values)
        original_audio = exporter.convert(payload.original_specs, payload.min_max_values)
        visualizer = SpectrogramVisualizer(SpectrogramPlotConfig(cmap="magma", vmin=0.0, vmax=1.0))

        run = self._create_run(
            len(payload.original_specs),
            run_name=self._run_name_for_payload(payload),
            seed=getattr(self.config, "seed", 42),
        )
        run.audio_paths.extend(
            exporter.save_signals(
                self._reconstruction_audio_groups(original_audio, reconstructed_audio),
                run.dir("audio"),
            )
        )
        run.spectrogram_paths.extend(
            visualizer.save_comparisons(payload.original_specs, payload.reconstructed_specs, run.dir("spectrograms"))
        )

        if payload.codebook_indices is not None:
            code_dir = run.dir("codebook")
            run.spectrogram_paths.extend(
                visualizer.save_code_indices(payload.codebook_indices, code_dir, "codebook_indices")
            )
            if payload.codebook_num_embeddings is not None:
                visualizer.save_code_histogram(
                    payload.codebook_indices,
                    int(payload.codebook_num_embeddings),
                    os.path.join(code_dir, "codebook_histogram.png"),
                )

        if payload.save_sampled_paths_array:
            run.save_array("sampled_file_paths.npy", np.asarray(payload.sampled_paths, dtype=object))

        metadata = {"config": self._config_dict()}
        if payload.include_sampled_paths_metadata:
            metadata["sampled_paths"] = payload.sampled_paths
        metadata.update(payload.metadata)
        run.save_json("metadata.json", run.metadata_payload(metadata))
        return run.result()


class PriorEvaluator(BaseEvaluator):
    """!
    @brief Shared run flow for prior sampling evaluators.
    """

    def _run_prior_artifacts(self, device: torch.device) -> Optional[EvaluationResult]:
        """!
        @brief Optional external artifact runner for complex prior evaluators.
        @return EvaluationResult when the subclass handles the full run, otherwise None.
        """
        return None

    def _produce_generation(self, device: torch.device) -> GenerationPayload:
        """!
        @brief Produce generated spectrograms and indices for common artifact export.
        """
        raise NotImplementedError

    def _save_generation_indices(self, visualizer: SpectrogramVisualizer, run: EvaluationRun, payload: GenerationPayload) -> None:
        """!
        @brief Save generated index arrays.
        """
        if not payload.indices:
            return
        index_dir = run.dir("indices")
        for name, indices in payload.indices.items():
            run.spectrogram_paths.extend(visualizer.save_code_indices(indices, index_dir, name))

    def run(self) -> EvaluationResult:
        """!
        @brief Run prior sampling evaluation and save common artifacts.
        @return Evaluation artifact paths.
        """
        device = self._device()
        external_result = self._run_prior_artifacts(device)
        if external_result is not None:
            return external_result

        payload = self._produce_generation(device)
        exporter = self._exporter(payload.run_config)
        signals = exporter.convert(payload.specs, payload.min_max_values)

        run = self._create_run(
            len(payload.specs),
            run_name=self.run_name,
            seed=getattr(self.config, "seed", 42),
        )
        visualizer = SpectrogramVisualizer()
        run.audio_paths.extend(exporter.save_signals({"generated": signals}, run.dir("audio")))
        run.spectrogram_paths.extend(
            visualizer.save_batch(
                payload.specs,
                run.dir("spectrograms"),
                payload.spectrogram_prefix,
                payload.spectrogram_title,
            )
        )
        self._save_generation_indices(visualizer, run, payload)

        metadata = {"config": self._config_dict()}
        metadata.update(payload.metadata)
        run.save_json("metadata.json", run.metadata_payload(metadata))
        return run.result()
