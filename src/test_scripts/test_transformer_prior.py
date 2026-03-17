from typing import Tuple, Optional

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from test_scripts.hierarchical_pixelcnn_common import resolve_model_paths
from utils import load_config

from modeling.torch.transformer_prior_conditioned import TransformerPriorConditioned


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
        h, w = int(grid[0]), int(grid[1])
        vis_dir = os.path.join(save_dir, 'visualizations', name)
        os.makedirs(vis_dir, exist_ok=True)
        for i in range(indices.shape[0]):
            img = indices[i].reshape(h, w)
            plt.figure(figsize=(5, 4))
            plt.imshow(img, origin='lower', aspect='auto')
            plt.colorbar()
            plt.title(f'Generated {name.capitalize()} Codes {i}')
            plt.savefig(os.path.join(vis_dir, f'sample_{i}.png'))
            plt.close()


def test_transformer_prior(
    top_prior_path: str,
    middle_prior_path: str,
    bottom_prior_path: str,
    num_samples: int,
    temperature: float,
    top_k: int,
    weights_file: str,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    top_k_value = top_k if top_k > 0 else None

    print(f'Loading top Transformer prior from {top_prior_path}')
    top_prior, top_config, _ = load_transformer_prior('top', top_prior_path, device, weights_file)
    top_seq_len = int(top_config['model']['inferred_seq_lens']['top'])
    top_grid = top_config['model'].get('inferred_grids', {}).get('top')

    with torch.no_grad():
        top_tokens = top_prior.generate(
            batch_size=num_samples,
            start_tokens=None,
            seq_len=top_seq_len,
            temperature=temperature,
            top_k=top_k_value,
            device=device,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample top/middle/bottom VQ indices from trained Transformer priors')
    parser.add_argument('--top_prior', type=str, required=True, help='Path to top prior run directory, config, or .pth')
    parser.add_argument('--middle_prior', type=str, required=True, help='Path to middle prior run directory, config, or .pth')
    parser.add_argument('--bottom_prior', type=str, required=True, help='Path to bottom prior run directory, config, or .pth')
    parser.add_argument('--weights_file', type=str, default='best_model.pth')
    parser.add_argument('--n_samples', type=int, default=6)
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--top_k', type=int, default=64)
    args = parser.parse_args()

    if args.temperature <= 0:
        raise ValueError(f'--temperature must be > 0, got {args.temperature}')
    if args.top_k < 0:
        raise ValueError(f'--top_k must be >= 0, got {args.top_k}')

    test_transformer_prior(
        top_prior_path=args.top_prior,
        middle_prior_path=args.middle_prior,
        bottom_prior_path=args.bottom_prior,
        num_samples=args.n_samples,
        temperature=args.temperature,
        top_k=args.top_k,
        weights_file=args.weights_file,
    )
