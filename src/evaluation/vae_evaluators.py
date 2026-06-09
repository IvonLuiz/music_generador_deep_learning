from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from evaluation.base_evaluators import ReconstructionPayload, VQVAEEvaluator
from evaluation.model_loading import ModelLoader
from generation.audio_inversion import DEFAULT_FIXED_MAX_DB, DEFAULT_FIXED_MIN_DB, AudioInversionConfig
from processing.preprocess_audio import TARGET_TIME_FRAMES
from train_scripts.jukebox_utils import parse_level
from utils import load_config


@dataclass
class SingleVQVAEReconstructionConfig:
    """!
    @brief Settings for single-level VQ-VAE reconstruction evaluation.
    """

    model_path: str
    weights_file: str = "best_model.pth"
    spectrograms_path: Optional[str] = None
    n_samples: int = 5
    seed: int = 42
    min_db: float = DEFAULT_FIXED_MIN_DB
    max_db: float = DEFAULT_FIXED_MAX_DB
    min_max_values_path: Optional[str] = None
    split: str = "validation"
    save_root: str = "samples/vqvae_reconstruction"


@dataclass
class HierarchicalVQVAEReconstructionConfig:
    """!
    @brief Settings for two-level VQ-VAE reconstruction evaluation.
    """

    model_path: str
    n_samples: int = 5
    seed: int = 42
    min_db: float = DEFAULT_FIXED_MIN_DB
    max_db: float = DEFAULT_FIXED_MAX_DB
    min_max_values_path: Optional[str] = None
    split: str = "validation"
    save_root: str = "samples/vq_vae_hierarchical_test"


@dataclass
class JukeboxVQVAEReconstructionConfig:
    """!
    @brief Settings for Jukebox VQ-VAE reconstruction evaluation.
    """

    model_path: str
    level: str = "bottom"
    weights_file: str = "best_model.pth"
    n_samples: int = 5
    target_time_frames: Optional[int] = None
    seed: int = 42
    min_db: float = DEFAULT_FIXED_MIN_DB
    max_db: float = DEFAULT_FIXED_MAX_DB
    min_max_values_path: Optional[str] = None
    save_root: str = "samples/jukebox_vqvae_maestro_test"
    audio_method: str = "gradient"
    split: str = "validation"


class SingleVQVAEReconstructionEvaluator(VQVAEEvaluator):
    """!
    @brief Evaluates reconstruction quality for a single-level VQ-VAE.
    """

    run_name = "single_vqvae"

    def _evaluate_reconstruction(self, device: torch.device) -> ReconstructionPayload:
        run_config = load_config(self._config_path_for(self.config.model_path))
        dataset_cfg = run_config.get("dataset", {})
        input_mode = str(dataset_cfg.get("input_mode", "spectrogram")).strip().lower()
        if input_mode == "audio" and self.config.spectrograms_path is None:
            sampled_specs, sampled_min_max, sampled_paths = self._sample_audio_specs(run_config, device)
            metadata = {"sample_source": "runtime_raw_audio", "split": self.config.split}
        else:
            specs_path = self.config.spectrograms_path or dataset_cfg.get("processed_path")
            if not specs_path:
                raise ValueError("A spectrogram path is required for VQ-VAE reconstruction evaluation.")
            target_frames = int(dataset_cfg.get("target_time_frames", TARGET_TIME_FRAMES))
            sampled_specs, sampled_paths = self._sample_npy_specs(specs_path, target_frames)
            sampled_min_max = self._min_max_for_paths(
                sampled_paths,
                specs_path,
                self.config.min_max_values_path,
                self.config.min_db,
                self.config.max_db,
            )
            metadata = {"sample_source": "precomputed_spectrograms"}

        model = ModelLoader.load_single_vqvae(self.config.model_path, device, self.config.weights_file)
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(sampled_specs).permute(0, 3, 1, 2).float().to(device)
            recon_out = model.reconstruct(x)
            recon = recon_out[0] if isinstance(recon_out, tuple) else recon_out
            recon_specs = recon.detach().cpu().permute(0, 2, 3, 1).numpy()

        return ReconstructionPayload(run_config, sampled_specs, recon_specs, sampled_min_max, sampled_paths, metadata=metadata)


class HierarchicalVQVAEReconstructionEvaluator(VQVAEEvaluator):
    """!
    @brief Evaluates reconstruction quality for a two-level VQ-VAE.
    """

    run_name = "hierarchical_vqvae"

    def _evaluate_reconstruction(self, device: torch.device) -> ReconstructionPayload:
        run_config = load_config(self._config_path_for(self.config.model_path))
        dataset_cfg = run_config.get("dataset", {})
        input_mode = str(dataset_cfg.get("input_mode", "spectrogram")).strip().lower()
        if input_mode == "audio":
            sampled_specs, sampled_min_max, sampled_paths = self._sample_audio_specs(run_config, device)
            metadata = {"sample_source": "runtime_raw_audio", "split": self.config.split}
        else:
            specs_path = dataset_cfg.get("processed_path")
            if not specs_path:
                raise ValueError("dataset.processed_path missing from VQ-VAE config.")
            target_frames = int(dataset_cfg.get("target_time_frames", TARGET_TIME_FRAMES))
            sampled_specs, sampled_paths = self._sample_npy_specs(specs_path, target_frames)
            sampled_min_max = self._min_max_for_paths(
                sampled_paths,
                specs_path,
                self.config.min_max_values_path,
                self.config.min_db,
                self.config.max_db,
            )
            metadata = {"sample_source": "precomputed_spectrograms"}

        model = ModelLoader.load_hierarchical_vqvae(self.config.model_path, device)
        x = torch.from_numpy(sampled_specs).permute(0, 3, 1, 2).float().to(device)
        model.eval()
        with torch.no_grad():
            x_recon, _, _ = model(x)
        recon_specs = x_recon.detach().cpu().permute(0, 2, 3, 1).numpy()

        return ReconstructionPayload(run_config, sampled_specs, recon_specs, sampled_min_max, sampled_paths, metadata=metadata)


class JukeboxVQVAEReconstructionEvaluator(VQVAEEvaluator):
    """!
    @brief Evaluates reconstruction and codebook usage for one Jukebox VQ-VAE level.
    """

    def __init__(self, config: JukeboxVQVAEReconstructionConfig, audio_config: Optional[AudioInversionConfig] = None):
        super().__init__(config, audio_config)
        self.level = parse_level(config.level)

    def _target_frames(self, run_config: dict) -> int:
        if self.config.target_time_frames is not None:
            return int(self.config.target_time_frames)
        profile = run_config.get("model", {}).get("level_profiles", {}).get(self.level, {})
        return int(profile.get("target_time_frames", run_config.get("dataset", {}).get("target_time_frames", TARGET_TIME_FRAMES)))

    def _reconstruction_audio_groups(self, original_signals, reconstructed_signals):
        return {"original": original_signals, "reconstructed": reconstructed_signals}

    def _run_name_for_payload(self, payload: ReconstructionPayload) -> str:
        return self.level

    def _evaluate_reconstruction(self, device: torch.device) -> ReconstructionPayload:
        model_ref = ModelLoader.model_reference(self.config.model_path)
        run_config = load_config(os.path.join(model_ref, "config.yaml"))
        target_frames = self._target_frames(run_config)

        dataset_cfg = run_config.get("dataset", {})
        input_mode = str(dataset_cfg.get("input_mode", "spectrogram")).strip().lower()
        if input_mode == "audio":
            sampled_specs, sampled_min_max, sampled_paths = self._sample_audio_specs(
                run_config,
                device,
                target_time_frames=target_frames,
            )
            metadata = {
                "target_time_frames": target_frames,
                "sample_source": "runtime_raw_audio",
                "split": self.config.split,
            }
        else:
            specs_path = dataset_cfg.get("processed_path")
            if not specs_path:
                raise ValueError("dataset.processed_path missing from Jukebox VQ-VAE config.")
            sampled_specs, sampled_paths = self._sample_npy_specs(specs_path, target_frames)
            sampled_min_max = self._min_max_for_paths(
                sampled_paths,
                specs_path,
                self.config.min_max_values_path,
                self.config.min_db,
                self.config.max_db,
            )
            metadata = {"target_time_frames": target_frames, "sample_source": "precomputed_spectrograms"}

        model = ModelLoader.load_jukebox_vqvae(model_ref, self.level, device, self.config.weights_file)
        x = torch.from_numpy(sampled_specs).permute(0, 3, 1, 2).float().to(device)
        model.eval()
        with torch.no_grad():
            x_recon, _, _ = model(x)
            indices = model.encode_to_indices(x)
        recon_specs = x_recon.detach().cpu().permute(0, 2, 3, 1).numpy()

        return ReconstructionPayload(
            run_config=run_config,
            original_specs=sampled_specs,
            reconstructed_specs=recon_specs,
            min_max_values=sampled_min_max,
            sampled_paths=sampled_paths,
            metadata=metadata,
            codebook_indices=indices.detach().cpu().numpy().astype(np.int64),
            codebook_num_embeddings=int(model.vq.num_embeddings),
            save_sampled_paths_array=True,
            include_sampled_paths_metadata=False,
        )
