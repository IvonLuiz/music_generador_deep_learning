from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from modeling.torch.pixel_cnn import ConditionalGatedPixelCNN
from modeling.torch.pixel_cnn_hierarchical import HierarchicalCondGatedPixelCNN
from modeling.torch.vq_vae import vqvae_loss
from modeling.torch.vq_vae_hierarchical import VQ_VAE_Hierarchical
from utils import initialize_vqvae_model

from .common import DataBundle, get_training_cfg, make_sample_generator
from .engine import StepResult, TrainingAdapter


class SingleVQVAEAdapter(TrainingAdapter):
    """!
    @brief Adapter for the single-level Residual VQ-VAE training script.

    This adapter keeps the old single VQ-VAE loss behavior while delegating the
    epoch loop, checkpointing and plotting to TrainingEngine.
    """

    latest_filename = 'model.pth'
    checkpoint_prefix = 'model_epoch'
    best_filenames = ('best_model.pth',)

    def build_model(self, config: dict, data: DataBundle, device: torch.device) -> torch.nn.Module:
        """!
        @brief Build the single VQ-VAE using the existing initialize_vqvae_model helper.
        """
        return initialize_vqvae_model(config, device)

    def build_optimizer(self, model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
        """!
        @brief Build the Adam optimizer with the legacy VQ-VAE weight decay default.
        """
        training_cfg = get_training_cfg(config)
        return torch.optim.Adam(
            model.parameters(),
            lr=float(training_cfg['learning_rate']),
            weight_decay=float(training_cfg.get('weight_decay', 1e-5)),
        )

    def train_step(self, model: torch.nn.Module, batch, data: DataBundle) -> StepResult:
        """!
        @brief Compute reconstruction and vector-quantization losses.
        """
        x_hat, _z, vq_loss, codebook_loss, commitment_loss = model(batch)
        loss, recon_loss = vqvae_loss(batch, x_hat, vq_loss, variance=max(float(data.data_variance), 1e-6))
        return StepResult(
            loss=loss,
            batch_size=int(batch.shape[0]),
            metrics={
                'total': loss,
                'reconstruction': recon_loss,
                'vq': vq_loss,
                'codebook': codebook_loss,
                'commitment': commitment_loss,
            },
        )

    def checkpoint_extra_state(self, model: torch.nn.Module, data: DataBundle, config: dict) -> dict:
        """!
        @brief Store the variance used to scale the reconstruction loss.
        """
        return {'data_variance': data.data_variance}

    def create_sample_callback(self, model, data: DataBundle, run_dir: str, device: torch.device, config: dict):
        """!
        @brief Create VQ-VAE reconstruction samples when callbacks.save_samples is enabled.
        """
        dataset_cfg = config.get('dataset', {})
        sample_count = int(config.get('callbacks', {}).get('sample_count', 4))
        return make_sample_generator(model, data, run_dir, device, dataset_cfg, sample_count=sample_count)


class TwoLevelVQVAEAdapter(SingleVQVAEAdapter):
    """!
    @brief Adapter for the two-level hierarchical VQ-VAE.
    """

    def build_model(self, config: dict, data: DataBundle, device: torch.device) -> torch.nn.Module:
        """!
        @brief Build a VQ_VAE_Hierarchical model from config.model.
        """
        model_cfg = config['model']
        dataset_cfg = config.get('dataset', {})
        input_shape = (
            int(dataset_cfg.get('n_mels', 256)),
            int(dataset_cfg.get('target_time_frames', 256)),
            1,
        )
        model = VQ_VAE_Hierarchical(
            input_shape=input_shape,
            dim_bottom=int(model_cfg['dim_bottom']),
            dim_top=int(model_cfg['dim_top']),
            num_residual_layers=int(model_cfg['num_residual_layers']),
            num_embeddings_top=int(model_cfg['num_embeddings_top']),
            num_embeddings_bottom=int(model_cfg['num_embeddings_bottom']),
            beta=float(model_cfg['beta']),
        )
        return model.to(device)

    def train_step(self, model: torch.nn.Module, batch, data: DataBundle) -> StepResult:
        """!
        @brief Compute reconstruction plus top and bottom VQ losses.
        """
        reconstructions, total_vq_loss, vq_losses_details = model(batch)
        if len(vq_losses_details) != 2:
            raise ValueError(f'TwoLevelVQVAEAdapter expected 2 VQ levels, got {len(vq_losses_details)}')

        top_vq_loss, top_codebook_loss, top_commitment_loss = vq_losses_details[0]
        bottom_vq_loss, bottom_codebook_loss, bottom_commitment_loss = vq_losses_details[1]
        recon_loss = F.mse_loss(reconstructions, batch) / (2 * max(float(data.data_variance), 1e-6))
        loss = recon_loss + total_vq_loss
        return StepResult(
            loss=loss,
            batch_size=int(batch.shape[0]),
            metrics={
                'total': loss,
                'reconstruction': recon_loss,
                'vq_top': top_vq_loss,
                'vq_bottom': bottom_vq_loss,
                'codebook_top': top_codebook_loss,
                'codebook_bottom': bottom_codebook_loss,
                'commitment_top': top_commitment_loss,
                'commitment_bottom': bottom_commitment_loss,
            },
        )


class SinglePixelCNNAdapter(TrainingAdapter):
    """!
    @brief Adapter for PixelCNN priors trained on one quantized VQ-VAE level.
    """

    latest_filename = 'latest_model.pth'
    checkpoint_prefix = 'pixelcnn_epoch'
    best_filenames = ('best_model.pth', 'best_pixelcnn_model.pth')

    def run_subdir(self, config: dict):
        """!
        @brief Use the model name as an extra run subdirectory.
        """
        return config.get('model', {}).get('name')

    def build_model(self, config: dict, data: DataBundle, device: torch.device) -> torch.nn.Module:
        """!
        @brief Build a conditional gated PixelCNN with K inferred from data or config.
        """
        model_cfg = config['model']
        num_embeddings = _resolve_num_embeddings(model_cfg, data)
        model_cfg['K'] = int(num_embeddings)
        return ConditionalGatedPixelCNN(
            in_channels=1,
            hidden_channels=int(model_cfg['hidden_channels']),
            num_layers=int(model_cfg['num_layers']),
            kernel_size=int(model_cfg['kernel_size']),
            num_classes=int(num_embeddings),
            num_embeddings=int(num_embeddings),
        ).to(device)

    def build_optimizer(self, model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
        """!
        @brief Build the Adam optimizer used by the PixelCNN priors.
        """
        training_cfg = get_training_cfg(config)
        return torch.optim.Adam(model.parameters(), lr=float(training_cfg['learning_rate']))

    def train_step(self, model: torch.nn.Module, batch, data: DataBundle) -> StepResult:
        """!
        @brief Compute categorical cross-entropy over discrete codebook indices.
        """
        indices = _as_index_tensor(batch)
        logits = model(indices).squeeze(2)
        loss = F.cross_entropy(logits, indices.long())
        return StepResult(
            loss=loss,
            batch_size=int(indices.shape[0]),
            metrics={'total': loss},
        )


class TwoLevelPixelCNNAdapter(SinglePixelCNNAdapter):
    """!
    @brief Adapter for hierarchical PixelCNN priors over top and bottom codes.
    """

    best_filenames = ('best_model.pth',)
    checkpoint_prefix = 'model_epoch'

    def run_subdir(self, config: dict):
        """!
        @brief Keep hierarchical PixelCNN runs separated from single-level runs.
        """
        model_name = config.get('model', {}).get('name', 'pixelcnn')
        return f'{model_name}_hierarchical_pixelcnn'

    def build_model(self, config: dict, data: DataBundle, device: torch.device) -> torch.nn.Module:
        """!
        @brief Build top and bottom PixelCNN levels with bottom conditioned on top.
        """
        model_cfg = config.get('model', {})
        prior_cfg = config.get('priors', {})
        top_cfg = prior_cfg.get('top_prior', config.get('top_prior', {}))
        bottom_cfg = prior_cfg.get('bottom_prior', config.get('bottom_prior', {}))
        num_embeddings = data.num_embeddings or [
            int(top_cfg.get('num_embeddings', model_cfg.get('num_embeddings_top', 512))),
            int(bottom_cfg.get('num_embeddings', model_cfg.get('num_embeddings_bottom', 512))),
        ]
        input_size = data.input_size or [(32, 32), (64, 64)]
        return HierarchicalCondGatedPixelCNN(
            num_prior_levels=int(model_cfg.get('num_prior_levels', 2)),
            input_size=input_size,
            hidden_units=[int(top_cfg['hidden_channels']), int(bottom_cfg['hidden_channels'])],
            num_layers=[int(top_cfg['num_layers']), int(bottom_cfg['num_layers'])],
            conv_filter_size=[int(top_cfg['conv_filter_size']), int(bottom_cfg['conv_filter_size'])],
            dropout=[float(top_cfg.get('dropout_rate', 0.0)), float(bottom_cfg.get('dropout_rate', 0.0))],
            num_embeddings=[int(num_embeddings[0]), int(num_embeddings[1])],
            two_level_conditioning_mode=model_cfg.get('two_level_conditioning_mode', 'deconv'),
        ).to(device)

    def train_step(self, model: torch.nn.Module, batch, data: DataBundle) -> StepResult:
        """!
        @brief Compute independent top loss and conditioned bottom loss.
        """
        top_indices, bottom_indices = _split_two_level_batch(batch)
        logits_top = model(top_indices, level='top').squeeze(2)
        loss_top = F.cross_entropy(logits_top, top_indices.long())
        logits_bottom = model(bottom_indices, cond=top_indices, level='bottom').squeeze(2)
        loss_bottom = F.cross_entropy(logits_bottom, bottom_indices.long())
        loss = loss_top + loss_bottom
        return StepResult(
            loss=loss,
            batch_size=int(top_indices.shape[0]),
            metrics={
                'total': loss,
                'top': loss_top,
                'bottom': loss_bottom,
            },
        )


def _resolve_num_embeddings(model_cfg: dict, data: DataBundle) -> int:
    """!
    @brief Resolve PixelCNN vocabulary size from config first, then dataset metadata.
    """
    for key in ('K', 'num_embeddings'):
        if model_cfg.get(key) is not None:
            return int(model_cfg[key])
    if data.num_embeddings is not None:
        return int(data.num_embeddings)
    raise ValueError('Could not determine num_embeddings/K for PixelCNN.')


def _as_index_tensor(batch: torch.Tensor) -> torch.Tensor:
    """!
    @brief Convert PixelCNN batches to integer tensors shaped as index grids.
    """
    if isinstance(batch, (tuple, list)):
        batch = batch[0]
    if batch.ndim == 4 and batch.shape[1] == 1:
        batch = batch.squeeze(1)
    return batch.long()


def _split_two_level_batch(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """!
    @brief Unpack a hierarchical PixelCNN batch into top and bottom index grids.
    """
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        top_indices = batch[0]
        bottom_indices = batch[1]
    else:
        raise ValueError('Expected two-level PixelCNN batch as [top_indices, bottom_indices].')
    return _as_index_tensor(top_indices), _as_index_tensor(bottom_indices)
