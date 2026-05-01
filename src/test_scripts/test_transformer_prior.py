from typing import Tuple, Optional

import os
import sys
import argparse
import re
from datetime import datetime
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
import soundfile as sf

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from test_scripts.hierarchical_pixelcnn_common import resolve_model_paths
from utils import load_config
from generation.transformer_io_utils import (
    prepare_min_max_values,
    resolve_vqvae_config_path,
    save_decoded_spectrograms,
)

from modeling.torch.transformer_prior_conditioned import TransformerPriorConditioned
from generation.soundgenerator import SoundGenerator
from processing.preprocess_audio import SAMPLE_RATE, HOP_LENGTH, FRAME_SIZE, N_MELS
from train_scripts.jukebox_utils import load_jukebox_model


def _extract_num_embeddings(state_dict: dict) -> int:
    # Prefer output projection size because BOS only affects input token embedding size.
    for key in ('to_logits.weight', 'to_logits.weight_orig'):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    for key in ('token_embedding.weight', 'token_embedding.weight_orig'):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    raise KeyError('Could not infer num_embeddings from checkpoint state_dict')


def _extract_cond_num_embeddings(state_dict: dict) -> Optional[int]:
    for key in ('conditioner.token_embedding.weight', 'conditioner.token_embedding.weight_orig'):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    return None


def _extract_second_cond_num_embeddings(state_dict: dict) -> Optional[int]:
    for key in ('second_conditioner.token_embedding.weight', 'second_conditioner.token_embedding.weight_orig'):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    return None


def _extract_conditioner_block_count(state_dict: dict, prefix: str = 'conditioner') -> Optional[int]:
    pattern = re.compile(rf'^{re.escape(prefix)}\.layers\.(\d+)\.conv\.weight$')
    max_idx = -1
    for key in state_dict.keys():
        match = pattern.match(key)
        if not match:
            continue
        idx = int(match.group(1))
        if idx > max_idx:
            max_idx = idx
    if max_idx < 0:
        return None
    return max_idx + 1


def _extract_conditioner_width(state_dict: dict, prefix: str = 'conditioner') -> Optional[int]:
    for key in (f'{prefix}.token_embedding.weight', f'{prefix}.token_embedding.weight_orig'):
        if key in state_dict:
            return int(state_dict[key].shape[1])
    return None


def _extract_conditioner_kernel_size(state_dict: dict, prefix: str = 'conditioner') -> Optional[int]:
    for key in (f'{prefix}.layers.0.conv.weight', f'{prefix}.layers.0.conv.weight_orig'):
        if key in state_dict:
            return int(state_dict[key].shape[-1])
    return None


def _load_config_and_checkpoint(model_dir_or_file: str, weights_file: str):
    config_path, model_path = resolve_model_paths(model_dir_or_file, weights_file)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found at {config_path}')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found at {model_path}')

    config = load_config(config_path)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
    return config, state_dict, model_path


def load_transformer_prior(
    model_layer: str, model_dir_or_file: str,
    device: torch.device, weights_file: str = 'best_model.pth'
) -> Tuple[TransformerPriorConditioned, dict, str]:
    assert model_layer in ['top', 'middle', 'bottom'], f'model_layer must be one of "top", "middle", or "bottom", got {model_layer}'

    config, state_dict, model_path = _load_config_and_checkpoint(model_dir_or_file, weights_file)

    model_cfg = config.get('model', {})
    inferred_seq_lens = model_cfg.get('inferred_seq_lens', {})
    seq_len = int(inferred_seq_lens.get(model_layer, 0))
    if seq_len <= 0:
        raise ValueError(f'Missing model.inferred_seq_lens for level={model_layer}.')

    priors = config.get('priors', {})
    prior_cfg = priors.get(f'{model_layer}_prior') if isinstance(priors, dict) else None
    if prior_cfg is None:
        raise ValueError(f"Missing priors.{model_layer}_prior in config.")

    inferred_num_embeddings = _extract_num_embeddings(state_dict)
    num_embeddings_cfg = int(prior_cfg.get('num_embeddings', 0))
    num_embeddings = inferred_num_embeddings if num_embeddings_cfg <= 0 else num_embeddings_cfg

    # If config disagrees with checkpoint shape, trust checkpoint for load compatibility.
    if num_embeddings != inferred_num_embeddings:
        print(f'Warning: num_embeddings from config ({num_embeddings}) does not match checkpoint ({inferred_num_embeddings}). Using checkpoint value for loading model.')
        num_embeddings = inferred_num_embeddings

    use_bos_token = bool(prior_cfg.get('use_bos_token', False))
    token_embedding_weight = state_dict.get('token_embedding.weight')
    if token_embedding_weight is None:
        token_embedding_weight = state_dict.get('token_embedding.weight_orig')

    to_logits_weight = state_dict.get('to_logits.weight')
    if to_logits_weight is None:
        to_logits_weight = state_dict.get('to_logits.weight_orig')

    if token_embedding_weight is not None and to_logits_weight is not None:
        token_vocab = int(token_embedding_weight.shape[0])
        output_vocab = int(to_logits_weight.shape[0])
        if token_vocab == output_vocab + 1:
            use_bos_token = True
        elif token_vocab == output_vocab:
            use_bos_token = False

    max_time_steps_cfg = int(prior_cfg.get('max_time_steps', 0))
    max_time_steps = max_time_steps_cfg if max_time_steps_cfg > 0 else 100


    cond_num_embeddings = None
    upsample_stride = None
    second_cond_num_embeddings = None
    second_upsample_stride = None
    condition_on_top = bool(prior_cfg.get('condition_on_top', False)) if model_layer == 'bottom' else False
    if model_layer != 'top':
        cond_num_embeddings = int(prior_cfg.get('cond_num_embeddings', 0))
        if cond_num_embeddings <= 0:
            cond_num_embeddings = _extract_cond_num_embeddings(state_dict)
        cond_level = 'top' if model_layer == 'middle' else 'middle'
        inferred_stride = int(model_cfg.get('inferred_upsample_stride', 0))
        if inferred_stride > 0:
            upsample_stride = inferred_stride
        else:
            upper_len = int(inferred_seq_lens.get(cond_level, 0))
            if upper_len > 0 and seq_len % upper_len == 0:
                upsample_stride = seq_len // upper_len

        if model_layer == 'bottom':
            second_cond_num_embeddings = int(prior_cfg.get('second_cond_num_embeddings', 0))
            if second_cond_num_embeddings <= 0:
                second_cond_num_embeddings = _extract_second_cond_num_embeddings(state_dict)

            inferred_second_stride = int(model_cfg.get('inferred_second_upsample_stride', 0))
            if inferred_second_stride > 0:
                second_upsample_stride = inferred_second_stride
            else:
                top_len = int(inferred_seq_lens.get('top', 0))
                if top_len > 0 and seq_len % top_len == 0:
                    second_upsample_stride = seq_len // top_len

            # If checkpoint contains second conditioner, force-enable dual conditioning.
            if second_cond_num_embeddings is not None and second_upsample_stride is not None:
                condition_on_top = True

    conditioner_width_inferred = _extract_conditioner_width(state_dict, prefix='conditioner')
    conditioner_blocks_inferred = _extract_conditioner_block_count(state_dict, prefix='conditioner')
    conditioner_kernel_inferred = _extract_conditioner_kernel_size(state_dict, prefix='conditioner')

    conditioner_residual_block_width = int(
        prior_cfg.get(
            'conditioner_residual_block_width',
            conditioner_width_inferred if conditioner_width_inferred is not None else 1024,
        )
    )
    conditioner_residual_blocks = int(
        prior_cfg.get(
            'conditioner_residual_blocks',
            conditioner_blocks_inferred if conditioner_blocks_inferred is not None else 16,
        )
    )
    conditioner_kernel_size = int(
        prior_cfg.get(
            'conditioner_kernel_size',
            conditioner_kernel_inferred if conditioner_kernel_inferred is not None else 3,
        )
    )
    conditioner_conv_channels = int(
        prior_cfg.get('conditioner_conv_channels', conditioner_residual_block_width)
    )
    conditioner_dilation_growth_rate = int(prior_cfg.get('conditioner_dilation_growth_rate', 3))
    conditioner_dilation_cycle = int(prior_cfg.get('conditioner_dilation_cycle', 8))

    prior_transformer = TransformerPriorConditioned(
        num_embeddings=num_embeddings,
        model_dim=int(prior_cfg['model_dim']),
        num_heads=int(prior_cfg['num_heads']),
        num_layers=int(prior_cfg['num_layers']),
        dim_feedforward=int(prior_cfg['dim_feedforward']),
        max_seq_len=seq_len,
        block_len=int(prior_cfg.get('block_len', 16)),
        max_time_steps=max_time_steps,
        is_upsampler=model_layer != 'top',
        cond_num_embeddings=cond_num_embeddings if model_layer != 'top' else None,
        upsample_stride=upsample_stride if model_layer != 'top' else None,
        second_cond_num_embeddings=second_cond_num_embeddings if condition_on_top else None,
        second_upsample_stride=second_upsample_stride if condition_on_top else None,
        conditioner_residual_block_width=conditioner_residual_block_width,
        conditioner_residual_blocks=conditioner_residual_blocks,
        conditioner_kernel_size=conditioner_kernel_size,
        conditioner_conv_channels=conditioner_conv_channels,
        conditioner_dilation_growth_rate=conditioner_dilation_growth_rate,
        conditioner_dilation_cycle=conditioner_dilation_cycle,
        use_bos_token=use_bos_token,
        attention_qkv_ratio=float(prior_cfg.get('attention_qkv_ratio', 1.0)),
        dropout=float(prior_cfg.get('dropout', 0.1)),
    ).to(device)

    prior_transformer.load_state_dict(state_dict)
    prior_transformer.eval()

    return prior_transformer, config, model_path


def _save_indices(indices: np.ndarray, save_dir: str, name: str, grid: Optional[list]):
    path = os.path.join(save_dir, f'{name}_indices.npy')
    np.save(path, indices)
    print(f'Saved generated {name} indices to {path}')

    if isinstance(grid, list) and len(grid) == 2 and int(grid[0]) * int(grid[1]) == indices.shape[1]:
        # grid[0] is Time, grid[1] Frequency
        time_steps, freq_bins = int(grid[0]), int(grid[1])

        vis_dir = os.path.join(save_dir, 'visualizations', name)
        os.makedirs(vis_dir, exist_ok=True)
        for i in range(indices.shape[0]):
            # Reshape exactly as it was generated (Time, Freq), then transpose (.T) for the image (Freq, Time)
            img = indices[i].reshape(time_steps, freq_bins).T
            
            plt.figure(figsize=(5, 4))
            plt.imshow(img, origin='lower', aspect='auto')
            plt.colorbar()
            plt.title(f'Generated {name.capitalize()} Codes {i}')
            plt.savefig(os.path.join(vis_dir, f'sample_{i}.png'))
            plt.close()


def _decode_indices(
    vqvae,
    indices: torch.Tensor,
    grid: Optional[list],
) -> np.ndarray:
    """
    Decode VQ-VAE indices into spectrograms.

    @param vqvae: Loaded JukeboxVQVAE model.
    @param indices: numpy array of shape (B, T) with integer token indices.
    @param grid: [time_steps, freq_bins] matching the selected level grid.
    @return: numpy array of shape (B, H, W, 1) decoded spectrograms.
    """
    if indices.ndim != 2:
        raise ValueError(f'Expected indices shape (B, T), got {tuple(indices.shape)}')
    if not (isinstance(grid, list) and len(grid) == 2):
        raise ValueError('Bottom grid is required to reshape indices into (H, W)')

    device = next(vqvae.parameters()).device
    # grid[0] is now Time, grid[1] is now Frequency
    time_steps, freq_bins = int(grid[0]), int(grid[1])
    if time_steps * freq_bins != indices.shape[1]:
        raise ValueError(f'Grid {grid} does not match seq_len={indices.shape[1]}')

    idx_2d = indices.long().to(device)  # (B, T)
    idx_2d = idx_2d.view(idx_2d.shape[0], time_steps, freq_bins).transpose(1, 2).contiguous()  # (B, freq_bins, time_steps)
    
    vqvae.eval()
    with torch.no_grad():
        B, H, W = idx_2d.shape
        idx_flat = idx_2d.reshape(-1).to(device)   # (B * H * W)
        emb_flat = vqvae.vq.embedding[idx_flat]  # (B * H * W, D)
        emb = emb_flat.view(B, H, W, -1)  # (B, freq_bins, time_steps, D)
        z_q = emb.permute(0, 3, 1, 2).contiguous()  # (B, D, Freq, Time)
        x_hat = vqvae.decoder(z_q)
        if vqvae.activation_layer is not None:
            x_hat = vqvae.activation_layer(x_hat)

    return x_hat.detach().cpu().permute(0, 2, 3, 1).numpy()

def test_transformer_prior(
    top_prior_path: str,
    middle_prior_path: str,
    bottom_prior_path: str,
    bottom_vqvae_path: Optional[str],
    audio_method: str,
    num_samples: int,
    temperature: float,
    top_k: int,
    weights_file: str,
    decode_level: str = 'bottom',
    timing: Optional[torch.Tensor] = None,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    top_k_value = top_k if (top_k is not None and top_k > 0) else None

    print(f'Loading top Transformer prior from {top_prior_path}')
    top_prior, top_config, _ = load_transformer_prior('top', top_prior_path, device, weights_file)
    top_seq_len = int(top_config['model']['inferred_seq_lens']['top'])
    top_grid = top_config['model'].get('inferred_grids', {}).get('top')

    print(f'Loading middle Transformer prior from {middle_prior_path}')
    middle_prior, middle_config, _ = load_transformer_prior('middle', middle_prior_path, device, weights_file)
    middle_seq_len = int(middle_config['model']['inferred_seq_lens']['middle'])
    middle_grid = middle_config['model'].get('inferred_grids', {}).get('middle')

    print(f'Loading bottom Transformer prior from {bottom_prior_path}')
    bottom_prior, bottom_config, _ = load_transformer_prior('bottom', bottom_prior_path, device, weights_file)
    bottom_seq_len = int(bottom_config['model']['inferred_seq_lens']['bottom'])
    bottom_grid = bottom_config['model'].get('inferred_grids', {}).get('bottom')
    bottom_prior_cfg = bottom_config.get('priors', {}).get('bottom_prior', {}) if isinstance(bottom_config, dict) else {}
    bottom_condition_on_top = bool(bottom_prior_cfg.get('condition_on_top', True))

    vqvae_cfg = bottom_config.get('vqvae', {}) if isinstance(bottom_config, dict) else {}
    effective_bottom_vqvae = bottom_vqvae_path or vqvae_cfg.get('bottom_model_dir')
    effective_weights_file = weights_file or vqvae_cfg.get('weights_file', 'best_model.pth')

    # Default timing: start of a segment-length clip (fraction_elapsed=0, start_time=0).
    # total_duration is estimated from the bottom level's VQ-VAE segment length.
    if timing is None:
        dataset_cfg = bottom_config.get('dataset', {}) if isinstance(bottom_config, dict) else {}
        if effective_bottom_vqvae:
            vqvae_config_path = resolve_vqvae_config_path(effective_bottom_vqvae)
            vqvae_config = load_config(vqvae_config_path)
            dataset_cfg = vqvae_config.get('dataset', dataset_cfg) if isinstance(vqvae_config, dict) else dataset_cfg

        target_time_frames = int(dataset_cfg.get('target_time_frames', 2048))
        hop_length = int(dataset_cfg.get('hop_length', 256))
        sample_rate = int(dataset_cfg.get('sample_rate', 22050))
        segment_overlap = float(dataset_cfg.get('segment_overlap', 0.0))

        segment_samples = max(1, (target_time_frames - 1) * hop_length)
        segment_duration_s = segment_samples / sample_rate
        hop_samples = max(1, int(segment_samples * (1.0 - segment_overlap)))
        total_duration_s = segment_duration_s
        timing = torch.zeros((num_samples, 3), dtype=torch.float32, device=device)
        timing[:, 1] = total_duration_s

    print(f'Generating top-level indices...')
    with torch.no_grad():
        top_tokens = top_prior.generate(
            batch_size=num_samples,
            start_tokens=None,
            seq_len=top_seq_len,
            temperature=temperature,
            top_k=top_k_value,
            device=device,
            timing=timing,
        )

    print(f'Generating middle-level indices...')
    with torch.no_grad():
        middle_tokens = middle_prior.generate(
            batch_size=num_samples,
            start_tokens=None,
            upper_indices=top_tokens,
            seq_len=middle_seq_len,
            temperature=temperature,
            top_k=top_k_value,
            device=device,
            timing=timing,
        )

    print(f'Generating bottom-level indices...')
    with torch.no_grad():
        bottom_tokens = bottom_prior.generate(
            batch_size=num_samples,
            start_tokens=None,
            upper_indices=middle_tokens,
            second_upper_indices=top_tokens if bottom_condition_on_top else None,
            seq_len=bottom_seq_len,
            temperature=temperature,
            top_k=top_k_value,
            device=device,
            timing=timing,
        )

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join('samples', 'transformer_prior_maestro', current_time)
    os.makedirs(save_dir, exist_ok=True)

    _save_indices(top_tokens.cpu().numpy().astype(np.int64), save_dir, 'top', top_grid)
    _save_indices(middle_tokens.cpu().numpy().astype(np.int64), save_dir, 'middle', middle_grid)
    _save_indices(bottom_tokens.cpu().numpy().astype(np.int64), save_dir, 'bottom', bottom_grid)

    # del models from GPU before decoding and audio generation to free up memory for large tensors in decoding and Griffin-Lim
    del top_prior, middle_prior, bottom_prior
    torch.cuda.empty_cache()

    decode_level = str(decode_level).strip().lower()
    if decode_level not in ('bottom', 'middle', 'top'):
        raise ValueError(f'--decode_level must be one of "bottom", "middle", "top", got {decode_level}')

    if decode_level == 'bottom':
        decode_tokens = bottom_tokens
        decode_grid = bottom_grid
        effective_decode_vqvae = effective_bottom_vqvae
    elif decode_level == 'middle':
        decode_tokens = middle_tokens
        decode_grid = middle_grid
        effective_decode_vqvae = vqvae_cfg.get('middle_model_dir')
    else:
        decode_tokens = top_tokens
        decode_grid = top_grid
        effective_decode_vqvae = vqvae_cfg.get('top_model_dir')

    if not effective_decode_vqvae:
        raise ValueError(
            f'{decode_level} VQ-VAE path is required. Update vqvae.{decode_level}_model_dir in Transformer config.'
        )

    vqvae_config_path = resolve_vqvae_config_path(effective_decode_vqvae)
    vqvae_config = load_config(vqvae_config_path)
    resolved_dataset_cfg = vqvae_config.get('dataset', {}) if isinstance(vqvae_config, dict) else {}

    min_max_values_path = resolved_dataset_cfg.get('min_max_values_path')
    if not min_max_values_path:
        raise ValueError(f'Missing dataset.min_max_values_path in bottom VQ-VAE config: {vqvae_config_path}')

    sample_rate = int(resolved_dataset_cfg.get('sample_rate', SAMPLE_RATE))
    hop_length = int(resolved_dataset_cfg.get('hop_length', HOP_LENGTH))
    frame_size = int(resolved_dataset_cfg.get('frame_size', FRAME_SIZE))
    spectrograms_path = resolved_dataset_cfg.get('processed_path', '')
    spectrogram_type_cfg = resolved_dataset_cfg.get('spectrogram_type')
    spectrogram_type = str(spectrogram_type_cfg).strip().lower() if spectrogram_type_cfg else (
        'mel' if 'mel' in str(spectrograms_path).lower() else 'linear'
    )
    n_mels = int(resolved_dataset_cfg.get('n_mels', N_MELS))

    if effective_decode_vqvae is not None:
        print(f'Loading {decode_level} VQ-VAE from {effective_decode_vqvae}')
        vqvae = load_jukebox_model(effective_decode_vqvae, decode_level, device, effective_weights_file)

        if not os.path.exists(min_max_values_path):
            raise FileNotFoundError(f'min_max_values.pkl not found at {min_max_values_path}')

        with open(min_max_values_path, 'rb') as f:
            min_max_values = pickle.load(f)

        decoded_specs = _decode_indices(vqvae, decode_tokens, decode_grid)
        save_decoded_spectrograms(decoded_specs, save_dir)
        min_max_list = prepare_min_max_values(min_max_values, decoded_specs.shape[0])

        sound_generator = SoundGenerator(
            vqvae,
            hop_length=hop_length,
            sample_rate=sample_rate,
            n_fft=frame_size,
            spectrogram_type=spectrogram_type,
            n_mels=n_mels,
        )
        audio_signals = sound_generator.convert_spectrograms_to_audio(
            decoded_specs, min_max_list, method=audio_method
        )

        audio_dir = os.path.join(save_dir, 'audio')
        os.makedirs(audio_dir, exist_ok=True)
        for i, signal in enumerate(audio_signals):
            sf.write(os.path.join(audio_dir, f'sample_{i}.wav'), signal, sample_rate)
        print(f'Saved audio to {audio_dir}')
    else:
        print('No bottom VQ-VAE path provided, skipping spectrogram decoding and audio generation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample top/middle/bottom VQ indices from trained Transformer priors')
    parser.add_argument('--top_prior', type=str, required=True, help='Path to top prior run directory, config, or .pth')
    parser.add_argument('--middle_prior', type=str, required=True, help='Path to middle prior run directory, config, or .pth')
    parser.add_argument('--bottom_prior', type=str, required=True, help='Path to bottom prior run directory, config, or .pth')
    parser.add_argument('--bottom_vqvae', type=str, default=None, help='Path to bottom VQ-VAE run directory, config, or .pth')
    parser.add_argument('--audio_method', type=str, default='griffinlim', help='Audio inversion: griffinlim or istft')
    parser.add_argument('--weights_file', type=str, default='best_model.pth')
    parser.add_argument('--n_samples', type=int, default=6, help='Number of samples to generate (default: 6)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (default: 1.0)')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k filtering for sampling (0 or negative for no filtering)')
    parser.add_argument('--decode_level', type=str, default='bottom', choices=['bottom', 'middle', 'top'], help='Which level to decode to audio (default: bottom)')
    args = parser.parse_args()

    if args.temperature <= 0:
        raise ValueError(f'--temperature must be > 0, got {args.temperature}')
    if args.top_k is not None and args.top_k < 0:
        raise ValueError(f'--top_k must be >= 0, got {args.top_k}')

    if args.audio_method not in ('griffinlim', 'istft'):
        raise ValueError("--audio_method must be 'griffinlim' or 'istft'")

    test_transformer_prior(
        top_prior_path=args.top_prior,
        middle_prior_path=args.middle_prior,
        bottom_prior_path=args.bottom_prior,
        bottom_vqvae_path=args.bottom_vqvae,
        audio_method=args.audio_method,
        num_samples=args.n_samples,
        temperature=args.temperature,
        top_k=args.top_k,
        weights_file=args.weights_file,
        decode_level=args.decode_level,
    )
