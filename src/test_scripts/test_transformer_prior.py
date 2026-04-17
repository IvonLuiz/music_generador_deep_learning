from typing import Tuple, Optional

import os
import sys
import argparse
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

from modeling.torch.transformer_prior_conditioned import TransformerPriorConditioned
from generation.soundgenerator import SoundGenerator
from processing.preprocess_audio import SAMPLE_RATE, HOP_LENGTH
from train_scripts.jukebox_utils import load_jukebox_model


def _extract_num_embeddings(state_dict: dict) -> int:
    for key in ('token_embedding.weight', 'token_embedding.weight_orig'):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    raise KeyError('Could not infer num_embeddings from checkpoint state_dict')


def _extract_cond_num_embeddings(state_dict: dict) -> Optional[int]:
    for key in ('conditioner.token_embedding.weight', 'conditioner.token_embedding.weight_orig'):
        if key in state_dict:
            return int(state_dict[key].shape[0])
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

    num_embeddings = int(prior_cfg.get('num_embeddings', 0))
    if num_embeddings <= 0:
        num_embeddings = _extract_num_embeddings(state_dict)
    
    cond_num_embeddings = None
    upsample_stride = None
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

    prior_transformer = TransformerPriorConditioned(
        num_embeddings=num_embeddings,
        model_dim=int(prior_cfg['model_dim']),
        num_heads=int(prior_cfg['num_heads']),
        num_layers=int(prior_cfg['num_layers']),
        dim_feedforward=int(prior_cfg['dim_feedforward']),
        max_seq_len=seq_len,
        block_len=int(prior_cfg.get('block_len', 16)),
        is_upsampler=model_layer != 'top',
        cond_num_embeddings=cond_num_embeddings if model_layer != 'top' else None,
        upsample_stride=upsample_stride if model_layer != 'top' else None,
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


def _save_decoded_spectrograms(specs: np.ndarray, save_dir: str) -> None:
    spec_dir = os.path.join(save_dir, 'spectrograms')
    os.makedirs(spec_dir, exist_ok=True)

    np.save(os.path.join(spec_dir, 'bottom_decoded_specs.npy'), specs)

    for i in range(specs.shape[0]):
        img = specs[i, :, :, 0]
        plt.figure(figsize=(6, 4))
        plt.imshow(img, origin='lower', aspect='auto')
        plt.colorbar()
        plt.title(f'Decoded Bottom Spectrogram {i}')
        plt.tight_layout()
        plt.savefig(os.path.join(spec_dir, f'bottom_spec_{i:03d}.png'), dpi=150)
        plt.close()


def _prepare_min_max_values(min_max_values: object, count: int) -> list:
    if isinstance(min_max_values, dict):
        values = list(min_max_values.values())
    elif isinstance(min_max_values, list):
        values = min_max_values
    else:
        raise ValueError('min_max_values must be a dict or list')

    if not values:
        raise ValueError('min_max_values is empty')

    if len(values) >= count:
        return values[:count]

    repeats = (count + len(values) - 1) // len(values)
    tiled = (values * repeats)[:count]
    return tiled


def _decode_bottom_indices(
    vqvae,
    indices: torch.Tensor,
    grid: Optional[list],
    device: torch.device,
) -> np.ndarray:
    if indices.ndim != 2:
        raise ValueError(f'Expected indices shape (B, T), got {tuple(indices.shape)}')
    if not (isinstance(grid, list) and len(grid) == 2):
        raise ValueError('Bottom grid is required to reshape indices into (H, W)')

    # grid[0] is now Time, grid[1] is now Frequency
    time_steps, freq_bins = int(grid[0]), int(grid[1])
    
    if time_steps * freq_bins != indices.shape[1]:
        raise ValueError(f'Grid {grid} does not match seq_len={indices.shape[1]}')

    # View as (Batch, Time, Freq), then transpose to (Batch, Freq, Time)
    idx_2d = indices.view(indices.shape[0], time_steps, freq_bins).transpose(1, 2).contiguous().long().to(device)
    
    vqvae.eval()
    with torch.no_grad():
        emb = vqvae.vq.embedding[idx_2d]  # (B, Freq, Time, D)
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
    min_max_values_path: Optional[str],
    audio_method: str,
    num_samples: int,
    temperature: float,
    top_k: int,
    weights_file: str,
    time_ids: Optional[torch.Tensor] = None,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    top_k_value = top_k if (top_k is not None and top_k > 0) else None

    print(f'Loading top Transformer prior from {top_prior_path}')
    top_prior, top_config, _ = load_transformer_prior('top', top_prior_path, device, weights_file)
    top_seq_len = int(top_config['model']['inferred_seq_lens']['top'])
    top_grid = top_config['model'].get('inferred_grids', {}).get('top')
    
    if time_ids is None:
        time_ids=torch.zeros((num_samples, 1), dtype=torch.long, device=device)

    with torch.no_grad():
        top_tokens = top_prior.generate(
            batch_size=num_samples,
            start_tokens=None,
            seq_len=top_seq_len,
            temperature=temperature,
            top_k=top_k_value,
            device=device,
            time_id=time_ids,
        )

    print(f'Loading middle Transformer prior from {middle_prior_path}')
    middle_prior, middle_config, _ = load_transformer_prior('middle', middle_prior_path, device, weights_file)
    middle_seq_len = int(middle_config['model']['inferred_seq_lens']['middle'])
    middle_grid = middle_config['model'].get('inferred_grids', {}).get('middle')

    with torch.no_grad():
        middle_tokens = middle_prior.generate(
            batch_size=num_samples,
            start_tokens=None,
            upper_indices=top_tokens,
            seq_len=middle_seq_len,
            temperature=temperature,
            top_k=top_k_value,
            device=device,
        )

    print(f'Loading bottom Transformer prior from {bottom_prior_path}')
    bottom_prior, bottom_config, _ = load_transformer_prior('bottom', bottom_prior_path, device, weights_file)
    bottom_seq_len = int(bottom_config['model']['inferred_seq_lens']['bottom'])
    bottom_grid = bottom_config['model'].get('inferred_grids', {}).get('bottom')

    with torch.no_grad():
        bottom_tokens = bottom_prior.generate(
            batch_size=num_samples,
            start_tokens=None,
            upper_indices=middle_tokens,
            seq_len=bottom_seq_len,
            temperature=temperature,
            top_k=top_k_value,
            device=device,
        )

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join('samples', 'transformer_hierarchical_generated', current_time)
    os.makedirs(save_dir, exist_ok=True)

    _save_indices(top_tokens.cpu().numpy().astype(np.int64), save_dir, 'top', top_grid)
    _save_indices(middle_tokens.cpu().numpy().astype(np.int64), save_dir, 'middle', middle_grid)
    _save_indices(bottom_tokens.cpu().numpy().astype(np.int64), save_dir, 'bottom', bottom_grid)

    vqvae_cfg = bottom_config.get('vqvae', {}) if isinstance(bottom_config, dict) else {}
    effective_bottom_vqvae = bottom_vqvae_path or vqvae_cfg.get('bottom_model_dir')
    effective_weights_file = weights_file or vqvae_cfg.get('weights_file', 'best_model.pth')
    min_max_values_path = bottom_config['dataset']['min_max_values_path']

    if effective_bottom_vqvae is not None:
        print(f'Loading bottom VQ-VAE from {effective_bottom_vqvae}')
        vqvae = load_jukebox_model(effective_bottom_vqvae, 'bottom', device, effective_weights_file)

        if not os.path.exists(min_max_values_path):
            raise FileNotFoundError(f'min_max_values.pkl not found at {min_max_values_path}')

        with open(min_max_values_path, 'rb') as f:
            min_max_values = pickle.load(f)

        decoded_specs = _decode_bottom_indices(vqvae, bottom_tokens, bottom_grid, device)
        _save_decoded_spectrograms(decoded_specs, save_dir)
        min_max_list = _prepare_min_max_values(min_max_values, decoded_specs.shape[0])

        sound_generator = SoundGenerator(vqvae, hop_length=HOP_LENGTH)
        audio_signals = sound_generator.convert_spectrograms_to_audio(
            decoded_specs, min_max_list, method=audio_method
        )

        audio_dir = os.path.join(save_dir, 'audio')
        os.makedirs(audio_dir, exist_ok=True)
        for i, signal in enumerate(audio_signals):
            sf.write(os.path.join(audio_dir, f'sample_{i}.wav'), signal, SAMPLE_RATE)
        print(f'Saved audio to {audio_dir}')
    else:
        print('No bottom VQ-VAE path provided, skipping spectrogram decoding and audio generation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample top/middle/bottom VQ indices from trained Transformer priors')
    parser.add_argument('--top_prior', type=str, required=True, help='Path to top prior run directory, config, or .pth')
    parser.add_argument('--middle_prior', type=str, required=True, help='Path to middle prior run directory, config, or .pth')
    parser.add_argument('--bottom_prior', type=str, required=True, help='Path to bottom prior run directory, config, or .pth')
    parser.add_argument('--bottom_vqvae', type=str, default=None, help='Path to bottom VQ-VAE run directory, config, or .pth')
    parser.add_argument('--min_max_values', type=str, default=None, help='Path to min_max_values.pkl (optional)')
    parser.add_argument('--audio_method', type=str, default='griffinlim', help='Audio inversion: griffinlim or istft')
    parser.add_argument('--weights_file', type=str, default='best_model.pth')
    parser.add_argument('--n_samples', type=int, default=6, help='Number of samples to generate (default: 6)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (default: 1.0)')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k filtering for sampling (0 or negative for no filtering)')
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
        min_max_values_path=args.min_max_values,
        audio_method=args.audio_method,
        num_samples=args.n_samples,
        temperature=args.temperature,
        top_k=args.top_k,
        weights_file=args.weights_file,
    )
