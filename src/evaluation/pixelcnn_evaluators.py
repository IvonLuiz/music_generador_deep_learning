from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from evaluation.base_evaluators import GenerationPayload, PriorEvaluator
from evaluation.model_loading import ModelLoader
from generation.audio_inversion import DEFAULT_FIXED_MAX_DB, DEFAULT_FIXED_MIN_DB
from processing.preprocess_audio import N_MELS, TARGET_TIME_FRAMES
from utils import load_config


@dataclass
class SinglePixelCNNSamplingConfig:
    """!
    @brief Settings for single-level PixelCNN sampling evaluation.
    """

    pixelcnn_path: str
    vqvae_path: str
    n_samples: int = 5
    min_db: float = DEFAULT_FIXED_MIN_DB
    max_db: float = DEFAULT_FIXED_MAX_DB
    save_root: str = "samples/pixelcnn_generated"


@dataclass
class HierarchicalPixelCNNSamplingConfig:
    """!
    @brief Settings for hierarchical PixelCNN sampling evaluation.
    """

    pixelcnn_path: str
    vqvae_path: str
    n_samples: int = 3
    min_db: float = DEFAULT_FIXED_MIN_DB
    max_db: float = DEFAULT_FIXED_MAX_DB
    save_root: str = "samples/pixelcnn_hierarchical_generated"


class SinglePixelCNNSamplingEvaluator(PriorEvaluator):
    """!
    @brief Samples a single-level PixelCNN and decodes through a VQ-VAE.
    """

    run_name = "single_pixelcnn"

    @staticmethod
    def generate_samples(pixelcnn_model, num_samples, latent_shape, device):
        """!
        @brief Autoregressively sample a single-level PixelCNN.
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
        """
        vqvae_model.eval()
        with torch.no_grad():
            z_q = F.embedding(indices, vqvae_model.vq.embedding)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
            x_hat = vqvae_model.decoder(z_q)
        return x_hat.permute(0, 2, 3, 1).cpu().numpy()

    def _produce_generation(self, device: torch.device) -> GenerationPayload:
        vqvae = ModelLoader.load_single_vqvae(self.config.vqvae_path, device)
        pixelcnn = ModelLoader.load_single_pixelcnn(self.config.pixelcnn_path, device, int(vqvae.vq.num_embeddings))
        run_config = load_config(self._config_path_for(self.config.vqvae_path))
        dataset_cfg = run_config.get("dataset", {})
        n_mels = int(dataset_cfg.get("n_mels", N_MELS))
        target_frames = int(dataset_cfg.get("target_time_frames", TARGET_TIME_FRAMES))
        dummy = torch.zeros((1, 1, n_mels, target_frames), device=device)

        with torch.no_grad():
            latent_shape = tuple(vqvae.encoder(dummy).shape[2:])
        indices = self.generate_samples(pixelcnn, int(self.config.n_samples), latent_shape, device)

        specs = self.decode_indices(indices, vqvae)

        return GenerationPayload(
            run_config=run_config,
            specs=specs,
            indices={"indices": indices.cpu().numpy()},
            metadata={"latent_shape": list(latent_shape)},
            spectrogram_prefix="generated_specs",
            spectrogram_title="Generated spectrogram",
        )


class HierarchicalPixelCNNSamplingEvaluator(PriorEvaluator):
    """!
    @brief Samples a hierarchical PixelCNN and decodes through a two-level VQ-VAE.
    """

    run_name = "hierarchical_pixelcnn"

    @staticmethod
    def decode_hierarchical(top_indices, bottom_indices, vqvae_model):
        """!
        @brief Decode top/bottom hierarchical indices into spectrograms.
        """
        vqvae_model.eval()
        with torch.no_grad():
            z_top = F.embedding(top_indices, vqvae_model.vq_top.embedding).permute(0, 3, 1, 2).contiguous()
            z_bottom = F.embedding(bottom_indices, vqvae_model.vq_bottom.embedding).permute(0, 3, 1, 2).contiguous()
            decoded_top = vqvae_model.decoder_top(z_top)
            x_recon = vqvae_model.decoder_bottom(torch.cat([z_bottom, decoded_top], dim=1))
        return torch.sigmoid(x_recon).permute(0, 2, 3, 1).cpu().numpy()

    def _produce_generation(self, device: torch.device) -> GenerationPayload:
        vqvae = ModelLoader.load_hierarchical_vqvae(self.config.vqvae_path, device)
        pixelcnn, _ = ModelLoader.load_hierarchical_pixelcnn_model(self.config.pixelcnn_path, device)
        run_config = load_config(self._config_path_for(self.config.vqvae_path))
        dataset_cfg = run_config.get("dataset", {})
        n_mels = int(dataset_cfg.get("n_mels", N_MELS))
        target_frames = int(dataset_cfg.get("target_time_frames", TARGET_TIME_FRAMES))
        dummy = torch.zeros((1, 1, n_mels, target_frames), device=device)

        with torch.no_grad():
            enc_bottom = vqvae.encoder_bottom(dummy)
            enc_top = vqvae.encoder_top(enc_bottom)
            top_shape = tuple(vqvae.pre_vq_conv_top(enc_top).shape[2:])
            bottom_shape = tuple(vqvae.pre_vq_conv_bottom(enc_bottom).shape[2:])
            print(
                "Sampling hierarchical PixelCNN tokens: "
                f"top_shape={top_shape}, bottom_shape={bottom_shape}, n_samples={self.config.n_samples}",
                flush=True,
            )
            top_indices = pixelcnn.generate(shape=(self.config.n_samples, 1, top_shape[0], top_shape[1]), level="top").squeeze(1)
            bottom_indices = pixelcnn.generate(
                shape=(self.config.n_samples, 1, bottom_shape[0], bottom_shape[1]),
                cond=top_indices,
                level="bottom",
            ).squeeze(1)

        specs = self.decode_hierarchical(top_indices, bottom_indices, vqvae)

        return GenerationPayload(
            run_config=run_config,
            specs=specs,
            indices={
                "top_indices": top_indices.cpu().numpy(),
                "bottom_indices": bottom_indices.cpu().numpy(),
            },
            metadata={"top_shape": list(top_shape), "bottom_shape": list(bottom_shape)},
            spectrogram_prefix="generated_specs",
            spectrogram_title="Generated hierarchical spectrogram",
        )
