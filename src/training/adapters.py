from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.torch.jukebox_vq_vae import JukeboxVQVAE
from modeling.torch.pixel_cnn import ConditionalGatedPixelCNN
from modeling.torch.pixel_cnn_hierarchical import HierarchicalCondGatedPixelCNN
from modeling.torch.transformer_prior_conditioned import TransformerPriorConditioned
from modeling.torch.vq_vae import vqvae_loss
from modeling.torch.vq_vae_hierarchical import VQ_VAE_Hierarchical
from utils import initialize_vqvae_model

from .common import DataBundle, get_training_cfg, make_sample_generator
from .engine import StepResult, TrainingAdapter


LEVEL_TO_PRIOR_CFG = {'top': 'top_prior', 'middle': 'middle_prior', 'bottom': 'bottom_prior'}
COND_LEVEL = {'top': None, 'middle': 'top', 'bottom': 'middle'}
SECOND_COND_LEVEL = {'top': None, 'middle': None, 'bottom': 'top'}


class SingleVQVAEAdapter(TrainingAdapter):
    """!
    @brief Adapter for the single-level Residual VQ-VAE training script.

    This adapter keeps the old single VQ-VAE loss behavior while delegating the
    epoch loop, checkpointing and plotting to TrainingEngine.
    """

    def build_model(self, config: dict, data: DataBundle, device: torch.device) -> torch.nn.Module:
        """!
        @brief Build the single VQ-VAE using the existing initialize_vqvae_model helper.
        """
        return initialize_vqvae_model(config, device)

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

    def run_subdir(self, config: dict):
        """!
        @brief Use the model name as an extra run subdirectory.
        """
        return config.get('model', {}).get('name')

    def build_model(self, config: dict, data: DataBundle, device: torch.device) -> torch.nn.Module:
        """!
        @brief Build a conditional gated PixelCNN with K from config.
        """
        model_cfg = config['model']
        num_embeddings = int(model_cfg['K'])
        model_cfg['K'] = int(num_embeddings)
        return ConditionalGatedPixelCNN(
            in_channels=1,
            hidden_channels=int(model_cfg['hidden_channels']),
            num_layers=int(model_cfg['num_layers']),
            kernel_size=int(model_cfg['kernel_size']),
            num_classes=int(num_embeddings),
            num_embeddings=int(num_embeddings),
        ).to(device)

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
        num_embeddings = [
            int(top_cfg['num_embeddings']),
            int(bottom_cfg['num_embeddings']),
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


class JukeboxVQVAEAdapter(TrainingAdapter):
    """!
    @brief Adapter for one bottom/middle/top Jukebox VQ-VAE level.
    """

    def run_subdir(self, config: dict):
        model_name = config.get('model', {}).get('name', 'jukebox_vqvae')
        level = config.get('task', {}).get('level', config.get('model', {}).get('selected_level', 'bottom'))
        return f'{model_name}_{level}'

    def build_model(self, config: dict, data: DataBundle, device: torch.device) -> torch.nn.Module:
        model_cfg = config['model']
        level = config.get('task', {}).get('level', model_cfg.get('selected_level', 'bottom'))
        level_profile = model_cfg.get('level_profiles', {}).get(level)
        if not level_profile:
            raise ValueError(f"Missing model.level_profiles.{level} in Jukebox config.")
        activation_name = str(model_cfg.get('activation', '')).lower()
        activation_layer = nn.Sigmoid() if activation_name == 'sigmoid' else None
        required = [
            'input_channels', 'num_embeddings', 'embedding_dim', 'beta', 'conv_type',
            'dilation_growth_rate', 'channel_growth', 'ema_decay', 'epsilon',
            'restart_threshold',
        ]
        missing = [key for key in required if key not in model_cfg]
        if missing:
            raise ValueError(f"Missing required Jukebox model config parameters: {', '.join(missing)}")
        return JukeboxVQVAE(
            input_channels=int(model_cfg['input_channels']),
            hidden_dim=int(level_profile['hidden_dim']),
            levels=int(level_profile['levels']),
            num_residual_layers=int(level_profile.get('num_residual_layers', 4)),
            num_embeddings=int(model_cfg['num_embeddings']),
            embedding_dim=int(model_cfg['embedding_dim']),
            beta=float(model_cfg['beta']),
            conv_type=int(model_cfg['conv_type']),
            activation_layer=activation_layer,
            dilation_growth_rate=int(model_cfg['dilation_growth_rate']),
            channel_growth=int(model_cfg['channel_growth']),
            ema_decay=float(model_cfg['ema_decay']),
            epsilon=float(model_cfg['epsilon']),
            restart_threshold=float(model_cfg['restart_threshold']),
        ).to(device)

    def train_step(self, model: torch.nn.Module, batch, data: DataBundle) -> StepResult:
        reconstructions, total_vq_loss, vq_losses_details = model(batch)
        vq_loss, codebook_loss, commitment_loss = vq_losses_details[0]
        recon_loss = F.mse_loss(reconstructions, batch) / (2 * max(float(data.data_variance), 1e-6))
        loss = recon_loss + total_vq_loss
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
        level = config.get('task', {}).get('level', config.get('model', {}).get('selected_level', 'bottom'))
        return {
            'data_variance': data.data_variance,
            'selected_level': level,
            'level_profile': config.get('model', {}).get('level_profiles', {}).get(level, {}),
        }

    def create_sample_callback(self, model, data: DataBundle, run_dir: str, device: torch.device, config: dict):
        dataset_cfg = config.get('dataset', {})
        sample_count = int(config.get('callbacks', {}).get('sample_count', 4))
        return make_sample_generator(model, data, run_dir, device, dataset_cfg, sample_count=sample_count)


class JukeboxTransformerPriorAdapter(TrainingAdapter):
    """!
    @brief Adapter for Jukebox top/middle/bottom Transformer priors.
    """

    def run_subdir(self, config: dict):
        model_name = config.get('model', {}).get('name', 'jukebox')
        level = config.get('task', {}).get('level', config.get('model', {}).get('selected_level', 'top'))
        return f'{model_name}_{level}_transformer_prior'

    def build_model(self, config: dict, data: DataBundle, device: torch.device) -> torch.nn.Module:
        selected_level = config.get('task', {}).get('level', config.get('model', {}).get('selected_level', 'top'))
        prior_cfg = _get_prior_cfg(config, LEVEL_TO_PRIOR_CFG[selected_level])
        conditioning_cfg = config.get('conditioning', {})
        key_cfg = conditioning_cfg.get('key', {})
        vqvae_cfg = config.get('vqvae', {})
        metadata = data.metadata
        vqvae_codebook_size = int(vqvae_cfg.get('codebook_size', 2048))
        is_upsampler = COND_LEVEL[selected_level] is not None

        condition_on_top = bool(prior_cfg.get('condition_on_top', False)) and selected_level == 'bottom'
        second_cond_block_len = metadata.get('second_cond_block_len') if condition_on_top else None
        second_upsample_stride = metadata.get('second_upsample_stride') if condition_on_top else None

        prior = TransformerPriorConditioned(
            num_embeddings=vqvae_codebook_size,
            model_dim=int(prior_cfg['model_dim']),
            num_heads=int(prior_cfg['num_heads']),
            num_layers=int(prior_cfg['num_layers']),
            dim_feedforward=int(prior_cfg['dim_feedforward']),
            max_seq_len=int(metadata['target_seq_len']),
            block_len=int(prior_cfg.get('block_len', 16)),
            max_time_steps=int(prior_cfg.get('max_time_steps', 500)),
            is_upsampler=is_upsampler,
            cond_num_embeddings=vqvae_codebook_size,
            cond_block_len=metadata.get('cond_block_len'),
            upsample_stride=metadata.get('upsample_stride'),
            second_cond_num_embeddings=vqvae_codebook_size,
            second_cond_block_len=second_cond_block_len,
            second_upsample_stride=second_upsample_stride,
            conditioner_residual_block_width=int(prior_cfg.get('conditioner_residual_block_width', 1024)),
            conditioner_residual_blocks=int(prior_cfg.get('conditioner_residual_blocks', 16)),
            conditioner_kernel_size=int(prior_cfg.get('conditioner_kernel_size', 3)),
            conditioner_conv_channels=int(prior_cfg.get('conditioner_conv_channels', 1024)),
            conditioner_dilation_growth_rate=int(prior_cfg.get('conditioner_dilation_growth_rate', 3)),
            conditioner_dilation_cycle=int(prior_cfg.get('conditioner_dilation_cycle', 8)),
            dropout=float(prior_cfg.get('dropout', 0.1)),
            attention_qkv_ratio=float(prior_cfg.get('attention_qkv_ratio', 1.0)),
            attention_pattern=prior_cfg.get('attention_pattern'),
            use_bos_token=bool(prior_cfg.get('use_bos_token', False)),
            use_start_embedding=bool(prior_cfg.get('use_start_embedding', False)),
            tie_input_output_embeddings=bool(prior_cfg.get('tie_input_output_embeddings', False)),
            use_timing_conditioning=bool(prior_cfg.get('use_timing_conditioning', True)),
            timing_num_bins=int(prior_cfg.get('timing_num_bins', 1024)),
            duration_num_bins=int(prior_cfg.get('duration_num_bins', 256)),
            timing_window_seconds=float(prior_cfg.get('timing_window_seconds', metadata['timing_window_seconds'])),
            timing_max_duration_seconds=float(prior_cfg.get('timing_max_duration_seconds', 3600.0)),
            timing_embedding_init_std=float(prior_cfg.get('timing_embedding_init_std', 0.02)),
            timing_embedding_scale=float(prior_cfg.get('timing_embedding_scale', 1.0)),
            use_key_conditioning=bool(key_cfg.get('enabled', False)),
            key_num_classes=int(key_cfg.get('num_classes', 25)),
            key_unknown_id=int(key_cfg.get('unknown_id', 24)),
            key_embedding_scale=float(key_cfg.get('embedding_scale', 1.0)),
            key_embedding_init_std=_optional_float(key_cfg.get('embedding_init_std')),
            use_2d_conditioner=bool(prior_cfg.get('use_2d_conditioner', True)),
            initialization_std=_optional_float(prior_cfg.get('initialization_std')),
            position_embedding_init_std=_optional_float(prior_cfg.get('position_embedding_init_std')),
            zero_init_biases=bool(prior_cfg.get('zero_init_biases', True)),
        ).to(device)
        return prior

    def build_optimizer(self, model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
        training_cfg = get_training_cfg(config)
        adam_beta2 = float(training_cfg.get('adam_beta2', 0.95))
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(training_cfg['learning_rate']),
            weight_decay=float(training_cfg.get('weight_decay', 0.01)),
            betas=(0.9, adam_beta2),
        )

    def build_scheduler(self, optimizer: torch.optim.Optimizer, config: dict, steps_per_epoch: int):
        training_cfg = get_training_cfg(config)
        scheduler_name = str(training_cfg.get('scheduler', 'onecycle')).strip().lower()
        if scheduler_name in ('none', 'off', 'disabled'):
            print('Scheduler: disabled')
            return None
        if scheduler_name != 'onecycle':
            raise ValueError(f"Unsupported scheduler '{scheduler_name}'. Expected one of: onecycle, none")
        total_steps = max(1, int(steps_per_epoch) * int(training_cfg['epochs']))
        print(f'Scheduler: onecycle (optimizer_steps_per_epoch={steps_per_epoch}, total_steps={total_steps})')
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(training_cfg['learning_rate']),
            total_steps=total_steps,
            pct_start=float(training_cfg.get('scheduler_pct_start', 0.05)),
            anneal_strategy=str(training_cfg.get('scheduler_anneal_strategy', 'cos')),
        )

    def autocast_dtype(self, device: torch.device):
        if device.type == 'cuda':
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return None

    def train_step(self, model: torch.nn.Module, batch, data: DataBundle) -> StepResult:
        target_seq, cond_seq, second_cond_seq, timing, timing_mask, key_ids = _prepare_transformer_batch(batch)
        loss = model.loss(
            target_seq,
            upper_indices=cond_seq,
            second_upper_indices=second_cond_seq,
            timing=timing,
            timing_mask=timing_mask,
            key_ids=key_ids,
        )
        return StepResult(loss=loss, batch_size=int(target_seq.shape[0]), metrics={'total': loss})

    def load_model_state(self, model: torch.nn.Module, checkpoint: dict, config: dict, device: torch.device) -> dict:
        partial = _load_model_state_compatibly(model, checkpoint['model_state'])
        if partial:
            print('Checkpoint architecture differed; optimizer and scheduler state will be reset.')
            return {'reset_optimizer': True, 'reset_scheduler': True}
        return {}

    def config_for_save(self, config: dict, data: DataBundle, model: torch.nn.Module) -> dict:
        selected_level = config.get('task', {}).get('level', config.get('model', {}).get('selected_level', 'top'))
        prior_cfg = _get_prior_cfg(config, LEVEL_TO_PRIOR_CFG[selected_level])
        metadata = data.metadata
        config.setdefault('model', {})['selected_level'] = selected_level
        config['model']['inferred_seq_lens'] = dict(metadata.get('seq_lens', {}))
        config['model']['inferred_grids'] = {
            key: [int(shape[0]), int(shape[1])]
            for key, shape in metadata.get('grid_shapes', {}).items()
        }
        config['model']['use_bos_token'] = bool(prior_cfg.get('use_bos_token', False))
        config['model']['use_start_embedding'] = bool(prior_cfg.get('use_start_embedding', False))
        config['model']['tie_input_output_embeddings'] = bool(prior_cfg.get('tie_input_output_embeddings', False))
        config['model']['attention_pattern'] = prior_cfg.get('attention_pattern', 'factored')
        config['model']['initialization_std'] = _optional_float(prior_cfg.get('initialization_std'))
        config['model']['position_embedding_init_std'] = _optional_float(prior_cfg.get('position_embedding_init_std'))
        config['model']['zero_init_biases'] = bool(prior_cfg.get('zero_init_biases', True))
        config['model']['use_timing_conditioning'] = bool(prior_cfg.get('use_timing_conditioning', True))
        config['model']['use_key_conditioning'] = bool(config.get('conditioning', {}).get('key', {}).get('enabled', False))
        config['model']['key_num_classes'] = int(config.get('conditioning', {}).get('key', {}).get('num_classes', 25))
        config['model']['key_unknown_id'] = int(config.get('conditioning', {}).get('key', {}).get('unknown_id', 24))
        if config.get('conditioning', {}).get('key', {}).get('embedding_init_std') is not None:
            config['model']['key_embedding_init_std'] = float(
                config.get('conditioning', {}).get('key', {}).get('embedding_init_std')
            )
        config['model']['use_2d_conditioner'] = bool(prior_cfg.get('use_2d_conditioner', True))
        config['model']['timing_window_seconds'] = float(
            prior_cfg.get('timing_window_seconds', metadata['timing_window_seconds'])
        )
        config['model']['max_time_steps'] = int(prior_cfg.get('max_time_steps', 500))
        if metadata.get('upsample_stride') is not None:
            config['model']['inferred_upsample_stride'] = _serialize_stride(metadata['upsample_stride'])
        if metadata.get('second_upsample_stride') is not None:
            config['model']['inferred_second_upsample_stride'] = _serialize_stride(metadata['second_upsample_stride'])
        for key in ('cond_seq_len', 'cond_time_cols', 'cond_freq_bins',
                    'second_cond_seq_len', 'second_cond_time_cols', 'second_cond_freq_bins'):
            if int(metadata.get(key, 0)) > 0:
                config['model'][f'inferred_{key}'] = int(metadata[key])
        config.setdefault('dataset', {})['level_target_time_frames'] = dict(
            config.get('dataset', {}).get('level_target_time_frames', {})
        )
        return config


def _get_prior_cfg(config: dict, name: str) -> dict:
    priors = config.get('priors')
    if priors and name in priors:
        return priors[name]
    return config[name]


def _optional_float(value):
    if value is None:
        return None
    return float(value)


def _serialize_stride(stride):
    if isinstance(stride, (tuple, list)):
        if len(stride) != 2:
            raise ValueError(f'Expected stride tuple/list of length 2, got {stride}')
        return [int(stride[0]), int(stride[1])]
    return int(stride)


def _prepare_transformer_batch(batch):
    target_indices = batch[0].long()
    cond_indices = batch[1].long() if batch[1] is not None and batch[1].numel() > 0 else None
    second_cond_indices = batch[2].long() if batch[2] is not None and batch[2].numel() > 0 else None
    timing = batch[3].float()
    batch_metadata = batch[4] if len(batch) > 4 and isinstance(batch[4], dict) else {}
    key_ids = batch_metadata.get('key_id') if batch_metadata else None
    timing_mask = batch_metadata.get('timing_mask') if batch_metadata else None
    if key_ids is not None:
        key_ids = key_ids.long()
    if timing_mask is not None:
        timing_mask = timing_mask.bool()
    target_seq = target_indices.view(target_indices.shape[0], -1)
    cond_seq = cond_indices.view(cond_indices.shape[0], -1) if cond_indices is not None else None
    second_cond_seq = (
        second_cond_indices.view(second_cond_indices.shape[0], -1)
        if second_cond_indices is not None else None
    )
    return target_seq, cond_seq, second_cond_seq, timing, timing_mask, key_ids


def _load_model_state_compatibly(model: nn.Module, state_dict: dict) -> bool:
    try:
        model.load_state_dict(state_dict)
        return False
    except RuntimeError as exc:
        print(f'Strict checkpoint load failed; retrying with shape-compatible weights only: {exc}')

    current_state = model.state_dict()
    compatible_state = {}
    skipped = []
    unexpected = []
    for key, value in state_dict.items():
        if key not in current_state:
            unexpected.append(key)
            continue
        if tuple(current_state[key].shape) != tuple(value.shape):
            skipped.append(key)
            continue
        compatible_state[key] = value

    missing, unexpected_from_load = model.load_state_dict(compatible_state, strict=False)
    unexpected.extend(unexpected_from_load)
    if skipped:
        print(f'Skipped incompatible checkpoint weights ({len(skipped)} total): {skipped[:10]}')
    if missing:
        print(f'Model parameters initialized fresh ({len(missing)} total): {missing[:10]}')
    if unexpected:
        print(f'Ignored unexpected checkpoint keys ({len(unexpected)} total): {unexpected[:10]}')
    return True
