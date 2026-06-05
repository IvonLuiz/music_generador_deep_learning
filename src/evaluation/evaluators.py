from __future__ import annotations

import os
import pickle
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from evaluation.audio import AudioExporter
from evaluation.core import EvaluationOutputConfig, EvaluationResult, EvaluationRun
from evaluation.model_loading import ModelLoader
from evaluation.visualization import SpectrogramPlotConfig, SpectrogramVisualizer
from generation.audio_inversion import AudioGeometry, AudioInversionConfig
from processing.preprocess_audio import (
    FRAME_SIZE,
    HOP_LENGTH,
    N_MELS,
    SAMPLE_RATE,
    TARGET_TIME_FRAMES,
)
from train_scripts.jukebox_utils import parse_level
from utils import find_min_max_for_path, load_config, load_maestro, list_npy_files


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
    min_db: float = -80.0
    max_db: float = 0.0
    min_max_values_path: Optional[str] = None
    save_root: str = "samples/vqvae_reconstruction"


@dataclass
class HierarchicalVQVAEReconstructionConfig:
    """!
    @brief Settings for two-level VQ-VAE reconstruction evaluation.
    """

    model_path: str
    n_samples: int = 5
    seed: int = 42
    min_db: float = -80.0
    max_db: float = 0.0
    min_max_values_path: Optional[str] = None
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
    min_db: float = -80.0
    max_db: float = 0.0
    min_max_values_path: Optional[str] = None
    save_root: str = "samples/jukebox_vqvae_maestro_test"
    audio_method: str = "gradient"


@dataclass
class SinglePixelCNNSamplingConfig:
    """!
    @brief Settings for single-level PixelCNN sampling evaluation.
    """

    pixelcnn_path: str
    vqvae_path: str
    n_samples: int = 5
    min_db: float = -40.0
    max_db: float = 40.0
    save_root: str = "samples/pixelcnn_generated"


@dataclass
class HierarchicalPixelCNNSamplingConfig:
    """!
    @brief Settings for hierarchical PixelCNN sampling evaluation.
    """

    pixelcnn_path: str
    vqvae_path: str
    n_samples: int = 3
    min_db: float = -80.0
    max_db: float = 0.0
    save_root: str = "samples/pixelcnn_hierarchical_generated"


def _device() -> torch.device:
    """!
    @brief Return the evaluation device.
    @return CUDA device when available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def _sample_specs(spectrograms_path: str, target_time_frames: int, n_samples: int, seed: int):
    """!
    @brief Load and randomly sample precomputed spectrograms.
    @param spectrograms_path Directory containing spectrogram `.npy` files.
    @param target_time_frames Time frames requested by the model.
    @param n_samples Maximum number of samples.
    @param seed Sampling seed.
    @return Tuple `(sampled_specs, sampled_paths)`.
    """
    specs, file_paths = load_maestro(spectrograms_path, target_time_frames)
    count = min(int(n_samples), len(specs))
    if count <= 0:
        raise ValueError(f"No spectrogram samples found in {spectrograms_path}")
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(specs), count, replace=False)
    return specs[indices], [file_paths[i] for i in indices]


def _crop_or_pad_spectrogram(spectrogram: np.ndarray, target_time_frames: int) -> np.ndarray:
    """!
    @brief Crop or pad one 2D spectrogram to a target time length.
    @param spectrogram Spectrogram shaped `(F, T)`.
    @param target_time_frames Desired time-frame count.
    @return Cropped/padded spectrogram.
    """
    if spectrogram.shape[1] > target_time_frames:
        return spectrogram[:, :target_time_frames]
    if spectrogram.shape[1] < target_time_frames:
        return np.pad(spectrogram, ((0, 0), (0, target_time_frames - spectrogram.shape[1])), mode="constant")
    return spectrogram


def _sample_npy_specs(spectrograms_path: str, target_time_frames: int, n_samples: int, seed: int):
    """!
    @brief Sample raw `.npy` spectrogram files without min/max lookup.
    @param spectrograms_path Directory containing spectrogram `.npy` files.
    @param target_time_frames Desired time-frame count.
    @param n_samples Maximum number of files.
    @param seed Sampling seed.
    @return Tuple `(sampled_specs, sampled_paths)`.
    """
    paths = list_npy_files(spectrograms_path)
    count = min(int(n_samples), len(paths))
    if count <= 0:
        raise ValueError(f"No .npy spectrogram files found in {spectrograms_path}")
    rng = np.random.default_rng(int(seed))
    selected = [paths[i] for i in rng.choice(len(paths), count, replace=False)]
    specs = [_crop_or_pad_spectrogram(np.load(path), target_time_frames) for path in selected]
    return np.stack(specs, axis=0).astype(np.float32)[..., np.newaxis], selected


def _min_max_for_paths(
    sampled_paths: List[str],
    spectrograms_path: str,
    min_max_values_path: Optional[str],
    fallback_min_db: float,
    fallback_max_db: float,
) -> Optional[List[dict]]:
    """!
    @brief Load explicit min/max metadata for sampled spectrogram paths.
    @param sampled_paths Spectrogram paths selected for evaluation.
    @param spectrograms_path Dataset root used by path matching.
    @param min_max_values_path Optional path to `min_max_values.pkl`.
    @param fallback_min_db dB fallback for missing entries.
    @param fallback_max_db dB fallback for missing entries.
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


class SingleVQVAEReconstructionEvaluator:
    """!
    @brief Evaluates reconstruction quality for a single-level VQ-VAE.
    """

    def __init__(self, config: SingleVQVAEReconstructionConfig, audio_config: Optional[AudioInversionConfig] = None):
        """!
        @brief Initialize evaluator.
        @param config Evaluation settings.
        @param audio_config Audio inversion settings.
        """
        self.config = config
        self.audio_config = audio_config or AudioInversionConfig(
            method="gradient",
            use_fixed_db_scale=True,
            fixed_min_db=config.min_db,
            fixed_max_db=config.max_db,
        )

    def run(self) -> EvaluationResult:
        """!
        @brief Run reconstruction evaluation.
        @return Evaluation artifact paths.
        """
        device = _device()
        run_config = load_config(_config_path_for(self.config.model_path))
        dataset_cfg = run_config.get("dataset", {})
        specs_path = self.config.spectrograms_path or dataset_cfg.get("processed_path")
        if not specs_path:
            raise ValueError("A spectrogram path is required for VQ-VAE reconstruction evaluation.")
        target_frames = int(dataset_cfg.get("target_time_frames", TARGET_TIME_FRAMES))
        sampled_specs, sampled_paths = _sample_specs(specs_path, target_frames, self.config.n_samples, self.config.seed)
        sampled_min_max = _min_max_for_paths(
            sampled_paths,
            specs_path,
            self.config.min_max_values_path,
            self.config.min_db,
            self.config.max_db,
        )

        model = ModelLoader.load_single_vqvae(self.config.model_path, device, self.config.weights_file)
        with torch.no_grad():
            x = torch.from_numpy(sampled_specs).permute(0, 3, 1, 2).float().to(device)
            recon_out = model.reconstruct(x)
            recon = recon_out[0] if isinstance(recon_out, tuple) else recon_out
            recon_specs = recon.detach().cpu().permute(0, 2, 3, 1).numpy()

        exporter = AudioExporter(_audio_geometry_from_config(run_config), self.audio_config, autoencoder=model)
        recon_signals = exporter.convert(recon_specs, sampled_min_max)
        original_signals = exporter.convert(sampled_specs, sampled_min_max)
        run = EvaluationRun(EvaluationOutputConfig(self.config.save_root, "single_vqvae", len(sampled_specs), self.config.seed))
        visualizer = SpectrogramVisualizer(SpectrogramPlotConfig(cmap="viridis", vmin=0.0, vmax=1.0))
        run.audio_paths.extend(exporter.save_signals({"reconstructed": recon_signals, "original": original_signals}, run.dir("audio")))
        run.spectrogram_paths.extend(visualizer.save_comparisons(sampled_specs, recon_specs, run.dir("spectrograms")))
        run.save_json("metadata.json", run.metadata_payload({"config": asdict(self.config), "sampled_paths": sampled_paths}))
        return run.result()


class HierarchicalVQVAEReconstructionEvaluator:
    """!
    @brief Evaluates reconstruction quality for a two-level VQ-VAE.
    """

    def __init__(self, config: HierarchicalVQVAEReconstructionConfig, audio_config: Optional[AudioInversionConfig] = None):
        self.config = config
        self.audio_config = audio_config or AudioInversionConfig(
            method="gradient",
            use_fixed_db_scale=True,
            fixed_min_db=config.min_db,
            fixed_max_db=config.max_db,
        )

    def run(self) -> EvaluationResult:
        """!
        @brief Run hierarchical VQ-VAE reconstruction evaluation.
        @return Evaluation artifact paths.
        """
        device = _device()
        cfg = load_config(_config_path_for(self.config.model_path))
        specs_path = cfg.get("dataset", {}).get("processed_path")
        if not specs_path:
            raise ValueError("dataset.processed_path missing from VQ-VAE config.")
        sampled_specs, sampled_paths = _sample_specs(specs_path, TARGET_TIME_FRAMES, self.config.n_samples, self.config.seed)
        sampled_min_max = _min_max_for_paths(
            sampled_paths,
            specs_path,
            self.config.min_max_values_path,
            self.config.min_db,
            self.config.max_db,
        )
        model = ModelLoader.load_hierarchical_vqvae(self.config.model_path, device)
        x = torch.from_numpy(sampled_specs).permute(0, 3, 1, 2).float().to(device)
        model.eval()
        with torch.no_grad():
            x_recon, _, _ = model(x)
        recon_specs = x_recon.detach().cpu().permute(0, 2, 3, 1).numpy()

        exporter = AudioExporter(_audio_geometry_from_config(cfg), self.audio_config, autoencoder=model)
        recon_signals = exporter.convert(recon_specs, sampled_min_max)
        original_signals = exporter.convert(sampled_specs, sampled_min_max)
        run = EvaluationRun(EvaluationOutputConfig(self.config.save_root, "hierarchical_vqvae", len(sampled_specs), self.config.seed))
        visualizer = SpectrogramVisualizer(SpectrogramPlotConfig(cmap="viridis", vmin=0.0, vmax=1.0))
        run.audio_paths.extend(exporter.save_signals({"reconstructed": recon_signals, "original": original_signals}, run.dir("audio")))
        run.spectrogram_paths.extend(visualizer.save_comparisons(sampled_specs, recon_specs, run.dir("spectrograms")))
        run.save_json("metadata.json", run.metadata_payload({"config": asdict(self.config), "sampled_paths": sampled_paths}))
        return run.result()


class JukeboxVQVAEReconstructionEvaluator:
    """!
    @brief Evaluates reconstruction and codebook usage for one Jukebox VQ-VAE level.
    """

    def __init__(self, config: JukeboxVQVAEReconstructionConfig, audio_config: Optional[AudioInversionConfig] = None):
        self.config = config
        self.level = parse_level(config.level)
        self.audio_config = audio_config or AudioInversionConfig(
            method=config.audio_method,
            use_fixed_db_scale=True,
            fixed_min_db=config.min_db,
            fixed_max_db=config.max_db,
        )

    def _target_frames(self, run_config: dict) -> int:
        if self.config.target_time_frames is not None:
            return int(self.config.target_time_frames)
        profile = run_config.get("model", {}).get("level_profiles", {}).get(self.level, {})
        return int(profile.get("target_time_frames", run_config.get("dataset", {}).get("target_time_frames", TARGET_TIME_FRAMES)))

    def run(self) -> EvaluationResult:
        """!
        @brief Run Jukebox VQ-VAE reconstruction evaluation.
        @return Evaluation artifact paths.
        """
        device = _device()
        model_ref = ModelLoader.model_reference(self.config.model_path)
        cfg = load_config(os.path.join(model_ref, "config.yaml"))
        target_frames = self._target_frames(cfg)
        specs_path = cfg.get("dataset", {}).get("processed_path")
        if not specs_path:
            raise ValueError("dataset.processed_path missing from Jukebox VQ-VAE config.")
        sampled_specs, sampled_paths = _sample_npy_specs(specs_path, target_frames, self.config.n_samples, self.config.seed)
        sampled_min_max = _min_max_for_paths(
            sampled_paths,
            specs_path,
            self.config.min_max_values_path,
            self.config.min_db,
            self.config.max_db,
        )
        model = ModelLoader.load_jukebox_vqvae(model_ref, self.level, device, self.config.weights_file)
        x = torch.from_numpy(sampled_specs).permute(0, 3, 1, 2).float().to(device)
        model.eval()
        with torch.no_grad():
            x_recon, _, _ = model(x)
            indices = model.encode_to_indices(x)
        recon_specs = x_recon.detach().cpu().permute(0, 2, 3, 1).numpy()

        exporter = AudioExporter(_audio_geometry_from_config(cfg), self.audio_config, autoencoder=model)
        original_audio = exporter.convert(sampled_specs, sampled_min_max)
        recon_audio = exporter.convert(recon_specs, sampled_min_max)
        run = EvaluationRun(EvaluationOutputConfig(self.config.save_root, self.level, len(sampled_specs), self.config.seed))
        visualizer = SpectrogramVisualizer(SpectrogramPlotConfig(cmap="viridis", vmin=0.0, vmax=1.0))
        run.audio_paths.extend(exporter.save_signals({"original": original_audio, "reconstructed": recon_audio}, run.dir("audio")))
        run.spectrogram_paths.extend(visualizer.save_comparisons(sampled_specs, recon_specs, run.dir("spectrograms")))
        idx_np = indices.detach().cpu().numpy().astype(np.int64)
        code_dir = run.dir("codebook")
        run.spectrogram_paths.extend(visualizer.save_code_indices(idx_np, code_dir, "codebook_indices"))
        visualizer.save_code_histogram(idx_np, int(model.vq.num_embeddings), os.path.join(code_dir, "codebook_histogram.png"))
        run.save_array("sampled_file_paths.npy", np.asarray(sampled_paths, dtype=object))
        run.save_json("metadata.json", run.metadata_payload({"config": asdict(self.config), "target_time_frames": target_frames}))
        return run.result()


class SinglePixelCNNSamplingEvaluator:
    """!
    @brief Samples a single-level PixelCNN and decodes through a VQ-VAE.
    """

    def __init__(self, config: SinglePixelCNNSamplingConfig, audio_config: Optional[AudioInversionConfig] = None):
        self.config = config
        self.audio_config = audio_config or AudioInversionConfig(
            method="gradient",
            use_fixed_db_scale=True,
            fixed_min_db=config.min_db,
            fixed_max_db=config.max_db,
        )

    @staticmethod
    def generate_samples(pixelcnn_model, num_samples, latent_shape, device):
        """!
        @brief Autoregressively sample a single-level PixelCNN.
        @param pixelcnn_model Loaded PixelCNN model.
        @param num_samples Number of samples to generate.
        @param latent_shape Tuple `(H, W)`.
        @param device Torch device.
        @return Tensor of sampled indices shaped `(B, H, W)`.
        """
        pixelcnn_model.eval()
        height, width = latent_shape
        samples = torch.zeros((num_samples, height, width), dtype=torch.long, device=device)
        with torch.no_grad():
            for i in tqdm(range(height), desc="Generating rows"):
                for j in range(width):
                    logits = pixelcnn_model(samples)
                    if logits.ndim == 5:
                        logits = logits.squeeze(2)
                    probs = F.softmax(logits[:, :, i, j], dim=-1)
                    samples[:, i, j] = torch.multinomial(probs, 1).squeeze(1)
        return samples

    @staticmethod
    def decode_indices(indices, vqvae_model):
        """!
        @brief Decode VQ-VAE indices into spectrograms.
        @param indices Index tensor shaped `(B, H, W)`.
        @param vqvae_model Loaded VQ-VAE model.
        @return Spectrogram batch shaped `(B, F, T, 1)`.
        """
        vqvae_model.eval()
        with torch.no_grad():
            z_q = F.embedding(indices, vqvae_model.vq.embedding)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
            x_hat = vqvae_model.decoder(z_q)
        return x_hat.permute(0, 2, 3, 1).cpu().numpy()

    def run(self) -> EvaluationResult:
        """!
        @brief Run PixelCNN sampling evaluation.
        @return Evaluation artifact paths.
        """
        device = _device()
        vqvae = ModelLoader.load_single_vqvae(self.config.vqvae_path, device)
        pixelcnn = ModelLoader.load_single_pixelcnn(self.config.pixelcnn_path, device, int(vqvae.vq.num_embeddings))
        dummy = torch.zeros((1, 1, 256, TARGET_TIME_FRAMES), device=device)
        with torch.no_grad():
            latent_shape = tuple(vqvae.encoder(dummy).shape[2:])
        indices = self.generate_samples(pixelcnn, int(self.config.n_samples), latent_shape, device)
        specs = self.decode_indices(indices, vqvae)
        cfg = load_config(_config_path_for(self.config.vqvae_path))
        exporter = AudioExporter(_audio_geometry_from_config(cfg), self.audio_config, autoencoder=vqvae)
        signals = exporter.convert(specs)
        run = EvaluationRun(EvaluationOutputConfig(self.config.save_root, "single_pixelcnn", int(self.config.n_samples), 42))
        visualizer = SpectrogramVisualizer()
        run.audio_paths.extend(exporter.save_signals({"generated": signals}, run.dir("audio")))
        run.spectrogram_paths.extend(visualizer.save_batch(specs, run.dir("spectrograms"), "generated_specs", "Generated spectrogram"))
        run.spectrogram_paths.extend(visualizer.save_code_indices(indices.cpu().numpy(), run.dir("indices")))
        run.save_json("metadata.json", run.metadata_payload({"config": asdict(self.config), "latent_shape": list(latent_shape)}))
        return run.result()


class HierarchicalPixelCNNSamplingEvaluator:
    """!
    @brief Samples a hierarchical PixelCNN and decodes through a two-level VQ-VAE.
    """

    def __init__(self, config: HierarchicalPixelCNNSamplingConfig, audio_config: Optional[AudioInversionConfig] = None):
        self.config = config
        self.audio_config = audio_config or AudioInversionConfig(
            method="gradient",
            use_fixed_db_scale=True,
            fixed_min_db=config.min_db,
            fixed_max_db=config.max_db,
        )

    @staticmethod
    def decode_hierarchical(top_indices, bottom_indices, vqvae_model):
        """!
        @brief Decode top/bottom hierarchical indices into spectrograms.
        @param top_indices Top-level code indices.
        @param bottom_indices Bottom-level code indices.
        @param vqvae_model Loaded hierarchical VQ-VAE.
        @return Spectrogram batch shaped `(B, F, T, 1)`.
        """
        vqvae_model.eval()
        with torch.no_grad():
            z_top = F.embedding(top_indices, vqvae_model.vq_top.embedding).permute(0, 3, 1, 2).contiguous()
            z_bottom = F.embedding(bottom_indices, vqvae_model.vq_bottom.embedding).permute(0, 3, 1, 2).contiguous()
            decoded_top = vqvae_model.decoder_top(z_top)
            x_recon = vqvae_model.decoder_bottom(torch.cat([z_bottom, decoded_top], dim=1))
        return torch.sigmoid(x_recon).permute(0, 2, 3, 1).cpu().numpy()

    def run(self) -> EvaluationResult:
        """!
        @brief Run hierarchical PixelCNN sampling evaluation.
        @return Evaluation artifact paths.
        """
        device = _device()
        vqvae = ModelLoader.load_hierarchical_vqvae(self.config.vqvae_path, device)
        pixelcnn, _ = ModelLoader.load_hierarchical_pixelcnn_model(self.config.pixelcnn_path, device)
        dummy = torch.zeros((1, 1, TARGET_TIME_FRAMES, TARGET_TIME_FRAMES), device=device)
        with torch.no_grad():
            enc_bottom = vqvae.encoder_bottom(dummy)
            enc_top = vqvae.encoder_top(enc_bottom)
            top_shape = tuple(vqvae.pre_vq_conv_top(enc_top).shape[2:])
            bottom_shape = tuple(vqvae.pre_vq_conv_bottom(enc_bottom).shape[2:])
            top_indices = pixelcnn.generate(shape=(self.config.n_samples, 1, top_shape[0], top_shape[1]), level="top").squeeze(1)
            bottom_indices = pixelcnn.generate(
                shape=(self.config.n_samples, 1, bottom_shape[0], bottom_shape[1]),
                cond=top_indices,
                level="bottom",
            ).squeeze(1)
        specs = self.decode_hierarchical(top_indices, bottom_indices, vqvae)
        cfg = load_config(_config_path_for(self.config.vqvae_path))
        exporter = AudioExporter(_audio_geometry_from_config(cfg), self.audio_config, autoencoder=vqvae)
        signals = exporter.convert(specs)
        run = EvaluationRun(EvaluationOutputConfig(self.config.save_root, "hierarchical_pixelcnn", int(self.config.n_samples), 42))
        visualizer = SpectrogramVisualizer()
        run.audio_paths.extend(exporter.save_signals({"generated": signals}, run.dir("audio")))
        run.spectrogram_paths.extend(visualizer.save_batch(specs, run.dir("spectrograms"), "generated_specs", "Generated hierarchical spectrogram"))
        code_dir = run.dir("indices")
        run.spectrogram_paths.extend(visualizer.save_code_indices(top_indices.cpu().numpy(), code_dir, "top_indices"))
        run.spectrogram_paths.extend(visualizer.save_code_indices(bottom_indices.cpu().numpy(), code_dir, "bottom_indices"))
        run.save_json(
            "metadata.json",
            run.metadata_payload({"config": asdict(self.config), "top_shape": list(top_shape), "bottom_shape": list(bottom_shape)}),
        )
        return run.result()
