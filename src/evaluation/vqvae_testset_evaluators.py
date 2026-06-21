from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from evaluation.base_evaluators import BaseEvaluator
from evaluation.core import EvaluationResult
from evaluation.model_loading import ModelLoader
from processing.preprocess_audio import TARGET_TIME_FRAMES
from train_scripts.jukebox_utils import parse_level, split_paths_by_maestro_metadata
from utils import list_npy_files, load_config


@dataclass
class VQVAEModelTestSpec:
    """!
    @brief Description of one VQ-VAE checkpoint to evaluate on a dataset split.
    """

    name: str
    variant: str
    model_path: str
    weights_file: str = "best_model.pth"
    level: Optional[str] = None


@dataclass
class MaestroVQVAETestSetConfig:
    """!
    @brief Settings for comparing VQ-VAE implementations on a MAESTRO metadata split.
    """

    models: List[VQVAEModelTestSpec] = field(default_factory=list)
    split: str = "test"
    spectrograms_path: Optional[str] = None
    metadata_path: Optional[str] = None
    raw_path: Optional[str] = None
    batch_size: int = 8
    max_samples: Optional[int] = None
    seed: int = 42
    save_root: str = "samples/maestro_vqvae_testset"


class VQVAETestSetModelRunner:
    """!
    @brief Loads one VQ-VAE implementation and computes reconstruction MSE.
    """

    def __init__(self, spec: VQVAEModelTestSpec, device: torch.device):
        """!
        @brief Store model metadata and target device.
        @param spec Model checkpoint specification.
        @param device Device used for inference.
        """
        self.spec = spec
        self.device = device
        self.variant = str(spec.variant).strip().lower()
        self.level = parse_level(spec.level) if spec.level else None
        self.model_ref = ModelLoader.model_reference(spec.model_path)
        self.run_config = load_config(BaseEvaluator._config_path_for(self.model_ref))
        self.model = self._load_model()

    def dataset_config(self) -> dict:
        """!
        @brief Return the model dataset configuration.
        @return Dataset configuration dictionary.
        """
        return dict(self.run_config.get("dataset", {}))

    def target_time_frames(self) -> int:
        """!
        @brief Resolve the spectrogram time dimension expected by this model.
        @return Number of time frames used for evaluation inputs.
        """
        dataset_cfg = self.dataset_config()
        if self.variant == "jukebox":
            profile = self.run_config.get("model", {}).get("level_profiles", {}).get(self.level, {})
            return int(profile.get("target_time_frames", dataset_cfg.get("target_time_frames", TARGET_TIME_FRAMES)))
        return int(dataset_cfg.get("target_time_frames", TARGET_TIME_FRAMES))

    def evaluate(self, file_paths: List[str], batch_size: int) -> Dict:
        """!
        @brief Compute per-file and aggregate MSE for this model.
        @param file_paths Spectrogram files selected from the requested MAESTRO split.
        @param batch_size Number of spectrograms per inference batch.
        @return Dictionary with summary metrics and per-sample rows.
        """
        if not file_paths:
            raise ValueError(f"No files were provided for model {self.spec.name}.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}.")

        self.model.eval()
        target_frames = self.target_time_frames()
        rows = []
        with torch.no_grad():
            for start in range(0, len(file_paths), batch_size):
                batch_paths = file_paths[start:start + batch_size]
                batch_np = np.stack(
                    [self._load_spectrogram(path, target_frames) for path in batch_paths],
                    axis=0,
                ).astype(np.float32)
                x = torch.from_numpy(batch_np).permute(0, 3, 1, 2).float().to(self.device)
                recon = self._reconstruct(x)
                self._validate_reconstruction_shape(x, recon)
                mse = torch.mean((recon - x) ** 2, dim=(1, 2, 3)).detach().cpu().numpy()
                for idx, (path, value) in enumerate(zip(batch_paths, mse), start=start):
                    rows.append(
                        {
                            "model_name": self.spec.name,
                            "variant": self.variant,
                            "level": self.level or "",
                            "sample_index": idx,
                            "file_path": path,
                            "mse": float(value),
                            "rmse": float(np.sqrt(value)),
                        }
                    )

        values = np.asarray([row["mse"] for row in rows], dtype=np.float64)
        summary = {
            "model_name": self.spec.name,
            "variant": self.variant,
            "level": self.level,
            "model_path": self.spec.model_path,
            "weights_file": self.spec.weights_file,
            "target_time_frames": target_frames,
            "num_samples": int(values.size),
            "mse_mean": float(np.mean(values)),
            "mse_std": float(np.std(values)),
            "mse_median": float(np.median(values)),
            "mse_min": float(np.min(values)),
            "mse_max": float(np.max(values)),
            "rmse_mean": float(np.sqrt(np.mean(values))),
        }
        return {"summary": summary, "rows": rows}

    def _load_model(self):
        """!
        @brief Load the model implementation identified by the spec variant.
        @return Torch module ready for evaluation.
        """
        if self.variant == "single":
            return ModelLoader.load_single_vqvae(self.spec.model_path, self.device, self.spec.weights_file)
        if self.variant == "hierarchical":
            return ModelLoader.load_hierarchical_vqvae(self.spec.model_path, self.device)
        if self.variant == "jukebox":
            if not self.level:
                raise ValueError(f"Jukebox model {self.spec.name} requires a level.")
            return ModelLoader.load_jukebox_vqvae(self.model_ref, self.level, self.device, self.spec.weights_file)
        raise ValueError(f"Unsupported VQ-VAE variant '{self.spec.variant}'.")

    def _reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """!
        @brief Reconstruct one input batch for the configured model type.
        @param x Input batch in `(B, C, H, W)` format.
        @return Reconstructed batch in `(B, C, H, W)` format.
        """
        if hasattr(self.model, "reconstruct"):
            output = self.model.reconstruct(x)
            if isinstance(output, tuple):
                output = output[0]
            return output.float()
        output = self.model(x)
        if isinstance(output, tuple):
            output = output[0]
        return output.float()

    @staticmethod
    def _load_spectrogram(path: str, target_time_frames: int) -> np.ndarray:
        """!
        @brief Load, crop, or pad one normalized spectrogram file.
        @param path Path to a `.npy` spectrogram.
        @param target_time_frames Required time dimension.
        @return Spectrogram array in `(freq, time, channel)` layout.
        """
        spectrogram = np.load(path)
        if spectrogram.ndim == 3 and spectrogram.shape[-1] == 1:
            spectrogram = spectrogram[..., 0]
        if spectrogram.ndim != 2:
            raise ValueError(f"Expected a 2D spectrogram at {path}, got shape {spectrogram.shape}.")
        if spectrogram.shape[1] > target_time_frames:
            spectrogram = spectrogram[:, :target_time_frames]
        elif spectrogram.shape[1] < target_time_frames:
            pad_width = target_time_frames - spectrogram.shape[1]
            spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode="constant")
        return spectrogram[..., np.newaxis]

    def _validate_reconstruction_shape(self, x: torch.Tensor, recon: torch.Tensor) -> None:
        """!
        @brief Ensure MSE is computed only over matching tensors.
        @param x Original input tensor.
        @param recon Reconstructed tensor.
        """
        if tuple(x.shape) != tuple(recon.shape):
            raise ValueError(
                f"Model {self.spec.name} reconstructed shape {tuple(recon.shape)} "
                f"but input shape was {tuple(x.shape)}."
            )


class MaestroVQVAETestSetEvaluator(BaseEvaluator):
    """!
    @brief Compare one or more VQ-VAE checkpoints on the MAESTRO metadata test split.
    """

    run_name = "maestro_vqvae_testset"

    def run(self) -> EvaluationResult:
        """!
        @brief Run all configured model evaluations and save MSE reports.
        @return Evaluation result containing the output directory and metadata path.
        """
        split = str(self.config.split).strip().lower()
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"split must be one of train, validation, test; got {self.config.split}.")
        if not self.config.models:
            raise ValueError("At least one model spec is required.")

        device = self._device()
        run = self._create_run(
            int(self.config.max_samples or 0),
            run_name=self.run_name,
            seed=int(self.config.seed),
        )

        summaries = []
        all_rows = []
        for spec in self.config.models:
            runner = VQVAETestSetModelRunner(spec, device)
            file_paths = self._select_split_paths(runner.dataset_config())
            result = runner.evaluate(file_paths, int(self.config.batch_size))
            summary = dict(result["summary"])
            summary["split"] = split
            summary["spectrograms_path"] = self._spectrograms_path(runner.dataset_config())
            summaries.append(summary)
            all_rows.extend(result["rows"])

        ranked = sorted(summaries, key=lambda item: item["mse_mean"])
        self._save_csv(run.path("per_sample_mse.csv"), all_rows)
        run.save_json(
            "metrics.json",
            run.metadata_payload(
                {
                    "config": self._config_dict(),
                    "metric": "mean_squared_error",
                    "split": split,
                    "summaries": summaries,
                    "ranked_by_mse": ranked,
                }
            ),
        )
        return run.result()

    def _select_split_paths(self, dataset_cfg: dict) -> List[str]:
        """!
        @brief Select spectrogram paths belonging to the configured MAESTRO split.
        @param dataset_cfg Dataset config from a model run.
        @return Possibly capped and shuffled list of split paths.
        """
        spectrograms_path = self._spectrograms_path(dataset_cfg)
        if not spectrograms_path:
            raise ValueError("A spectrogram path is required. Set --spectrograms_path or dataset.processed_path.")
        all_paths = list_npy_files(spectrograms_path)
        if not all_paths:
            raise FileNotFoundError(f"No .npy spectrograms found at {spectrograms_path}.")

        split_cfg = dict(dataset_cfg)
        if self.config.metadata_path:
            split_cfg["metadata_path"] = self.config.metadata_path
        if self.config.raw_path:
            split_cfg["raw_path"] = self.config.raw_path
        train_paths, validation_paths, test_paths = split_paths_by_maestro_metadata(all_paths, split_cfg)
        paths_by_split = {
            "train": train_paths,
            "validation": validation_paths,
            "test": test_paths,
        }
        selected = list(paths_by_split[str(self.config.split).strip().lower()])
        if not selected:
            raise ValueError(f"The MAESTRO {self.config.split} split produced no spectrogram files.")

        rng = np.random.default_rng(int(self.config.seed))
        rng.shuffle(selected)
        if self.config.max_samples is not None:
            selected = selected[:int(self.config.max_samples)]
        return selected

    def _spectrograms_path(self, dataset_cfg: dict) -> Optional[str]:
        """!
        @brief Resolve the spectrogram root used for split evaluation.
        @param dataset_cfg Dataset config from a model run.
        @return Spectrogram directory or None.
        """
        return self.config.spectrograms_path or dataset_cfg.get("processed_path")

    @staticmethod
    def _save_csv(path: str, rows: List[Dict]) -> None:
        """!
        @brief Save per-sample MSE rows as CSV.
        @param path Destination CSV path.
        @param rows Per-sample metric rows.
        """
        fieldnames = ["model_name", "variant", "level", "sample_index", "file_path", "mse", "rmse"]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
