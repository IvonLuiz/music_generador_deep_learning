import os
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

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


def resolve_vqvae_min_max_values_path(vqvae_config: dict) -> str:
    """!
    @brief Resolve the min/max normalization file directly from a VQ-VAE config.
    @param vqvae_config Parsed VQ-VAE configuration dictionary.
    @return Path to the `min_max_values.pkl` file.
    @throws ValueError If the dataset config does not define `min_max_values_path`.
    """
    dataset_cfg = vqvae_config.get('dataset') or {}
    min_max_path = dataset_cfg.get('min_max_values_path')
    if not min_max_path:
        raise ValueError('dataset.min_max_values_path missing from VQ-VAE config.')
    return str(min_max_path)


def save_spectrogram_image(
    spec: np.ndarray,
    out_path: str,
    title: str,
    *,
    cmap: str = 'magma',
    figsize: tuple = (8, 4),
    dpi: int = 150,
    colorbar_label: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """!
    @brief Save a single spectrogram image to disk.
    @param spec Spectrogram array shaped either `(F, T)` or `(F, T, 1)`.
    @param out_path Output image path.
    @param title Figure title.
    @param cmap Matplotlib colormap name.
    @param figsize Figure size in inches.
    @param dpi Output image DPI.
    @param colorbar_label Optional label for the colorbar.
    @param vmin Optional lower display bound.
    @param vmax Optional upper display bound.
    """
    img = spec[:, :, 0] if spec.ndim == 3 else spec
    plt.figure(figsize=figsize)
    plt.imshow(img, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar_label:
        plt.colorbar(label=colorbar_label)
    else:
        plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def save_level_spectrograms(
    decoded_specs: np.ndarray,
    output_dir: str,
    level: str,
    *,
    root_subdir: str = 'spectrograms',
    include_level_subdir: bool = True,
    npy_filename: Optional[str] = None,
    filename_prefix: Optional[str] = None,
    title_template: Optional[str] = None,
    cmap: str = 'magma',
    figsize: tuple = (10, 4),
    dpi: int = 150,
    colorbar_label: str = 'Normalized amplitude',
) -> str:
    """!
    @brief Save a batch of decoded spectrograms for one hierarchy level.
    @param decoded_specs Spectrogram batch shaped `(B, F, T, 1)` or `(B, F, T)`.
    @param output_dir Base output directory.
    @param level Logical level name such as `top`, `middle`, or `bottom`.
    @param root_subdir Root folder under `output_dir` where spectrograms are stored.
    @param include_level_subdir Whether to create a per-level subdirectory under `root_subdir`.
    @param npy_filename Output filename for the saved `.npy` array.
    @param filename_prefix Prefix used for generated PNG files.
    @param title_template Figure title template. Use `{index}` for the sample index.
    @param cmap Matplotlib colormap name.
    @param figsize Figure size in inches.
    @param dpi Output image DPI.
    @param colorbar_label Label used for the colorbar.
    @return Path to the directory containing the saved spectrogram assets.
    """
    spectrogram_dir = os.path.join(output_dir, root_subdir, level) if include_level_subdir else os.path.join(output_dir, root_subdir)
    os.makedirs(spectrogram_dir, exist_ok=True)

    array_name = npy_filename or f'{level}_decoded_specs.npy'
    image_prefix = filename_prefix or f'{level}_spectrogram'
    title_fmt = title_template or f'{level.capitalize()} decoded spectrogram {{index}}'

    np.save(os.path.join(spectrogram_dir, array_name), decoded_specs)

    for i in range(decoded_specs.shape[0]):
        spec = decoded_specs[i, :, :, 0] if decoded_specs.ndim == 4 else decoded_specs[i]
        save_spectrogram_image(
            spec,
            os.path.join(spectrogram_dir, f'{image_prefix}_{i:03d}.png'),
            title_fmt.format(index=i),
            cmap=cmap,
            figsize=figsize,
            dpi=dpi,
            colorbar_label=colorbar_label,
        )

    return spectrogram_dir


def decode_jukebox_indices(
    vqvae,
    indices: torch.Tensor,
    grid: Optional[list] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """!
    @brief Decode Jukebox VQ-VAE code indices into spectrograms.
    @param vqvae Loaded Jukebox VQ-VAE model.
    @param indices Token indices shaped `(B, T)`, `(T, F)`, or `(B, F, T)`.
    @param grid Optional `[time_cols, freq_bins]` grid used when `indices` is flattened.
    @param device Optional target device. Defaults to the VQ-VAE device.
    @return Decoded spectrogram batch shaped `(B, F, T, 1)`.
    @throws ValueError If the indices shape/grid combination is invalid.
    """
    if indices.ndim not in (2, 3):
        raise ValueError(f'Expected indices shape (B, T) or (T, F), got {tuple(indices.shape)}')

    if device is None:
        device = next(vqvae.parameters()).device

    if indices.ndim == 2 and grid is not None:
        if not (isinstance(grid, list) and len(grid) == 2):
            raise ValueError('A [time_cols, freq_bins] grid is required to reshape flattened indices.')
        time_steps, freq_bins = int(grid[0]), int(grid[1])
        if time_steps * freq_bins != indices.shape[1]:
            raise ValueError(f'Grid {grid} does not match flattened seq_len={indices.shape[1]}')
        idx_2d = indices.long().to(device).view(indices.shape[0], time_steps, freq_bins).transpose(1, 2).contiguous()
    elif indices.ndim == 2:
        idx_2d = indices.unsqueeze(0).long().to(device).transpose(1, 2).contiguous()
    else:
        idx_2d = indices.long().to(device)
        if idx_2d.shape[1] < idx_2d.shape[2]:
            idx_2d = idx_2d.transpose(1, 2).contiguous()

    vqvae.eval()
    with torch.no_grad():
        batch_size, freq_bins, time_steps = idx_2d.shape            # (B, F, T)
        idx_flat = idx_2d.reshape(-1)                               # (B * F * T,)
        emb_flat = vqvae.vq.embedding[idx_flat]                     # (B * F * T, D)
        emb = emb_flat.view(batch_size, freq_bins, time_steps, -1)  # (B, F, T, D)
        z_q = emb.permute(0, 3, 1, 2).contiguous()                  # (B, D, F, T)
        x_hat = vqvae.decoder(z_q)                                  # (B, 1, F, T)
        if vqvae.activation_layer is not None:
            x_hat = vqvae.activation_layer(x_hat)

    return x_hat.detach().cpu().permute(0, 2, 3, 1).numpy()         # (B, F, T, 1)


def save_decoded_spectrograms(specs: np.ndarray, save_dir: str) -> None:
    """!
    @brief Save the final decoded bottom spectrogram batch for generation output.
    @param specs Bottom-level decoded spectrogram batch shaped `(B, F, T, 1)`.
    @param save_dir Generation output directory.
    """
    save_level_spectrograms(
        specs,
        save_dir,
        'bottom',
        root_subdir='spectrograms',
        include_level_subdir=False,
        npy_filename='bottom_decoded_specs.npy',
        filename_prefix='bottom_spec',
        title_template='Decoded Bottom Spectrogram {index}',
        cmap='magma',
        figsize=(6, 4),
        colorbar_label='Normalized amplitude',
    )
