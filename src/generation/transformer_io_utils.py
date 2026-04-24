import os
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from utils import load_config


def prepare_min_max_values(min_max_values: object, count: int) -> list:
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
    return (values * repeats)[:count]


def resolve_vqvae_config_path(model_dir_or_file: str) -> str:
    if os.path.isdir(model_dir_or_file):
        config_path = os.path.join(model_dir_or_file, 'config.yaml')
    elif os.path.isfile(model_dir_or_file):
        filename = os.path.basename(model_dir_or_file).lower()
        if filename in ('config.yaml', 'config.yml'):
            config_path = model_dir_or_file
        else:
            config_path = os.path.join(os.path.dirname(model_dir_or_file), 'config.yaml')
    else:
        raise FileNotFoundError(f'Bottom VQ-VAE reference does not exist: {model_dir_or_file}')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Bottom VQ-VAE config.yaml not found at {config_path}')

    return config_path


def resolve_min_max_values_path(bottom_config: dict, debug_fn: Optional[Callable[[str], None]] = None) -> str:
    def _normalize_candidate_path(path: Optional[str]) -> Optional[str]:
        if not path or not isinstance(path, str):
            return None
        return os.path.abspath(os.path.expanduser(path))

    candidates: List[str] = []

    direct_path = _normalize_candidate_path(bottom_config.get('dataset', {}).get('min_max_values_path'))
    if direct_path:
        candidates.append(direct_path)

    processed_path = _normalize_candidate_path(bottom_config.get('dataset', {}).get('processed_path'))
    if processed_path:
        candidates.append(os.path.join(processed_path, 'min_max_values.pkl'))

    bottom_model_dir = _normalize_candidate_path(bottom_config.get('vqvae', {}).get('bottom_model_dir'))
    if bottom_model_dir:
        bottom_vqvae_cfg = os.path.join(bottom_model_dir, 'config.yaml')
        if os.path.exists(bottom_vqvae_cfg):
            if debug_fn is not None:
                debug_fn(f'Loading bottom VQ-VAE config for min/max fallback: {bottom_vqvae_cfg}')
            vq_cfg = load_config(bottom_vqvae_cfg)
            vq_direct = _normalize_candidate_path(vq_cfg.get('dataset', {}).get('min_max_values_path'))
            if vq_direct:
                candidates.append(vq_direct)
            vq_processed = _normalize_candidate_path(vq_cfg.get('dataset', {}).get('processed_path'))
            if vq_processed:
                candidates.append(os.path.join(vq_processed, 'min_max_values.pkl'))

    seen = set()
    unique_candidates = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique_candidates.append(path)

    if debug_fn is not None:
        debug_fn('Min/max candidate paths checked:')
        for path in unique_candidates:
            debug_fn(f'  - {path}')

    for path in unique_candidates:
        if os.path.exists(path):
            if debug_fn is not None:
                debug_fn(f'Using min/max path: {path}')
            return path

    raise FileNotFoundError(
        'Could not resolve min_max_values.pkl. Checked dataset.min_max_values_path, '
        'dataset.processed_path/min_max_values.pkl, and bottom VQ-VAE config fallbacks.'
    )


def save_decoded_spectrograms(specs: np.ndarray, save_dir: str) -> None:
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
