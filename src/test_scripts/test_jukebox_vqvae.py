import argparse
import os
import pickle
import sys
from datetime import datetime
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generation.soundgenerator import SoundGenerator
from processing.preprocess_audio import HOP_LENGTH, MIN_MAX_VALUES_SAVE_DIR, SAMPLE_RATE, TARGET_TIME_FRAMES
from train_scripts.jukebox_utils import _parse_level, load_jukebox_model
from utils import find_min_max_for_path, load_maestro


def _resolve_min_max_values_path(path_override: Optional[str]) -> str:
    if path_override:
        return path_override
    return os.path.join(MIN_MAX_VALUES_SAVE_DIR, 'min_max_values.pkl')


def _resolve_model_reference(model_dir_or_file: str) -> str:
    if os.path.isfile(model_dir_or_file):
        name = os.path.basename(model_dir_or_file).lower()
        if name in ('config.yaml', 'config.yml'):
            return os.path.dirname(model_dir_or_file)
    return model_dir_or_file


def _load_min_max_values(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f'min_max_values.pkl not found at {path}')
    with open(path, 'rb') as f:
        return pickle.load(f)


def _extract_code_indices(model, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        z = model.encoder(x)
        z = model.pre_vq_conv(z)
        _, idx, _, _, _ = model.vq(z)
    return idx.long()


def _select_samples(
    specs: np.ndarray,
    file_paths: np.ndarray,
    min_max_values: dict,
    spectrograms_path: str,
    n_samples: int,
    seed: int,
) -> Tuple[np.ndarray, List[str], List[dict]]:
    if len(specs) == 0:
        raise ValueError('No spectrograms found in dataset.')

    n = min(n_samples, len(specs))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(specs), size=n, replace=False)

    sampled_specs = specs[chosen]
    sampled_paths = [str(file_paths[i]) for i in chosen]

    sampled_min_max: List[dict] = []
    for fp in sampled_paths:
        mm = find_min_max_for_path(fp, min_max_values, spectrograms_path)
        if mm is None:
            mm = {'min': -80.0, 'max': 0.0}
        sampled_min_max.append(mm)

    return sampled_specs, sampled_paths, sampled_min_max


def _save_audio(signals: List[np.ndarray], out_dir: str, prefix: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for i, signal in enumerate(signals):
        sf.write(os.path.join(out_dir, f'{prefix}_{i:03d}.wav'), signal, SAMPLE_RATE)


def _save_spectrogram_comparisons(original_specs: np.ndarray, recon_specs: np.ndarray, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'original_specs.npy'), original_specs)
    np.save(os.path.join(out_dir, 'reconstructed_specs.npy'), recon_specs)

    n = original_specs.shape[0]
    for i in range(n):
        orig = original_specs[i, :, :, 0]
        recon = recon_specs[i, :, :, 0]
        diff = np.abs(orig - recon)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        im0 = axes[0].imshow(orig, origin='lower', aspect='auto')
        axes[0].set_title(f'Original {i}')
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(recon, origin='lower', aspect='auto')
        axes[1].set_title(f'Reconstructed {i}')
        plt.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(diff, origin='lower', aspect='auto', cmap='hot')
        axes[2].set_title(f'Abs Diff {i}')
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'comparison_{i:03d}.png'), dpi=150)
        plt.close()


def _save_codebook_visualizations(indices: torch.Tensor, num_embeddings: int, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    idx_np = indices.cpu().numpy().astype(np.int64)
    np.save(os.path.join(out_dir, 'codebook_indices.npy'), idx_np)

    flat = idx_np.reshape(-1)
    hist = np.bincount(flat, minlength=num_embeddings)
    np.save(os.path.join(out_dir, 'codebook_histogram.npy'), hist)

    plt.figure(figsize=(12, 4))
    plt.bar(np.arange(num_embeddings), hist, width=1.0)
    plt.title('Codebook Usage Histogram')
    plt.xlabel('Code Index')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'codebook_histogram.png'), dpi=150)
    plt.close()

    for i in range(idx_np.shape[0]):
        plt.figure(figsize=(5, 4))
        plt.imshow(idx_np[i], origin='lower', aspect='auto')
        plt.title(f'Codebook Indices Sample {i}')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'indices_{i:03d}.png'), dpi=150)
        plt.close()


def test_jukebox_vqvae(
    model_dir_or_file: str,
    level: str,
    n_samples: int,
    target_time_frames: int,
    weights_file: str,
    min_max_values_path: Optional[str],
    seed: int,
    audio_method: str,
) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    level_name = _parse_level(level)
    model_ref = _resolve_model_reference(model_dir_or_file)

    print(f'Loading Jukebox VQ-VAE ({level_name}) from {model_ref}')
    model = load_jukebox_model(model_ref, level_name, device, weights_file)

    if os.path.isdir(model_ref):
        run_dir = model_ref
    else:
        run_dir = os.path.dirname(model_ref)

    config_path = os.path.join(run_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config not found at {config_path}')

    with open(config_path, 'r') as f:
        import yaml
        run_config = yaml.safe_load(f)

    dataset_cfg = run_config.get('dataset', {})
    spectrograms_path = dataset_cfg.get('processed_path', './data/processed/maestro_spectrograms_test/')

    mm_path = _resolve_min_max_values_path(min_max_values_path)
    min_max_values = _load_min_max_values(mm_path)

    print(f'Loading spectrogram dataset from {spectrograms_path}')
    specs, file_paths = load_maestro(spectrograms_path, target_time_frames)

    sampled_specs, sampled_paths, sampled_min_max = _select_samples(
        specs=specs,
        file_paths=file_paths,
        min_max_values=min_max_values,
        spectrograms_path=spectrograms_path,
        n_samples=n_samples,
        seed=seed,
    )

    x = torch.from_numpy(sampled_specs).permute(0, 3, 1, 2).float().to(device)

    model.eval()
    with torch.no_grad():
        x_recon, _, _ = model(x)

    indices = _extract_code_indices(model, x)
    recon_specs = x_recon.detach().cpu().permute(0, 2, 3, 1).numpy()

    generator = SoundGenerator(model, hop_length=HOP_LENGTH)
    print('Converting spectrograms to audio...')
    original_audio = generator.convert_spectrograms_to_audio(sampled_specs, sampled_min_max, method=audio_method)
    recon_audio = generator.convert_spectrograms_to_audio(recon_specs, sampled_min_max, method=audio_method)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join('samples', 'jukebox_vqvae_test', level_name, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    audio_dir = os.path.join(save_dir, 'audio')
    _save_audio(original_audio, audio_dir, 'original')
    _save_audio(recon_audio, audio_dir, 'reconstructed')

    spec_dir = os.path.join(save_dir, 'spectrograms')
    _save_spectrogram_comparisons(sampled_specs, recon_specs, spec_dir)

    code_dir = os.path.join(save_dir, 'codebook')
    _save_codebook_visualizations(indices, int(model.vq.num_embeddings), code_dir)

    np.save(os.path.join(save_dir, 'sampled_file_paths.npy'), np.array(sampled_paths, dtype=object))
    print(f'Saved test outputs to {save_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Jukebox VQ-VAE reconstruction and codebook encoding')
    parser.add_argument('--model_path', type=str, required=True, help='Path to Jukebox VQ-VAE run dir, config.yaml, or .pth')
    parser.add_argument('--level', type=str, default='bottom', help='Jukebox level: top, middle, bottom')
    parser.add_argument('--weights_file', type=str, default='best_model.pth', help='Weights file name if model_path is a dir')
    parser.add_argument('--n_samples', type=int, default=5, help='Number of samples to test')
    parser.add_argument('--target_time_frames', type=int, default=TARGET_TIME_FRAMES, help='Time frames used by load_maestro')
    parser.add_argument('--min_max_values', type=str, default=None, help='Optional path to min_max_values.pkl')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sample selection')
    parser.add_argument('--audio_method', type=str, default='griffinlim', help='Audio inversion: griffinlim or istft')
    args = parser.parse_args()

    if args.n_samples <= 0:
        raise ValueError(f'--n_samples must be > 0, got {args.n_samples}')
    if args.audio_method not in ('griffinlim', 'istft'):
        raise ValueError("--audio_method must be 'griffinlim' or 'istft'")

    test_jukebox_vqvae(
        model_dir_or_file=args.model_path,
        level=args.level,
        n_samples=args.n_samples,
        target_time_frames=args.target_time_frames,
        weights_file=args.weights_file,
        min_max_values_path=args.min_max_values,
        seed=args.seed,
        audio_method=args.audio_method,
    )
