import argparse
import json
import os
import pickle
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generation.soundgenerator import SoundGenerator
from train_scripts.jukebox_utils import load_jukebox_model
from utils import find_min_max_for_path, load_config


WINDOWED_MANIFEST = 'windowed_manifest.jsonl'
WINDOWED_CONFIG = 'windowed_quantization_config.json'


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _source_stem(path: str) -> str:
    return os.path.basename(path).replace('.npy', '')


def _legacy_source_basename(pt_path: str) -> str:
    return os.path.basename(pt_path).replace('_full_quantized.pt', '.npy')


def _resolve_vqvae_dir(
    quantized_root: str,
    level: str,
    explicit_vqvae_dir: Optional[str],
    transformer_config: Optional[str],
) -> str:
    if explicit_vqvae_dir:
        return explicit_vqvae_dir

    windowed_cfg_path = os.path.join(quantized_root, WINDOWED_CONFIG)
    if os.path.isfile(windowed_cfg_path):
        with open(windowed_cfg_path, 'r', encoding='utf-8') as f:
            windowed_cfg = json.load(f)
        vqvae_dir = ((windowed_cfg.get('vqvae_dirs') or {}).get(level))
        if vqvae_dir:
            return str(vqvae_dir)

    if transformer_config:
        config = load_config(transformer_config)
        vqvae_dir = (config.get('vqvae') or {}).get(f'{level}_model_dir')
        if vqvae_dir:
            return str(vqvae_dir)

    raise ValueError(
        f'Could not resolve VQ-VAE directory for level={level}. '
        'Pass --vqvae_model_dir or --transformer_config, or inspect a windowed quantized directory '
        'that contains windowed_quantization_config.json.'
    )


def _resolve_source_path(
    quantized_root: str,
    explicit_source_path: Optional[str],
    vqvae_config: dict,
) -> str:
    if explicit_source_path:
        return explicit_source_path

    windowed_cfg_path = os.path.join(quantized_root, WINDOWED_CONFIG)
    if os.path.isfile(windowed_cfg_path):
        with open(windowed_cfg_path, 'r', encoding='utf-8') as f:
            windowed_cfg = json.load(f)
        source_path = windowed_cfg.get('source_path')
        if source_path:
            return str(source_path)

    source_path = (vqvae_config.get('dataset') or {}).get('processed_path')
    if source_path:
        return str(source_path)

    raise ValueError(
        'Could not resolve processed spectrogram source path. '
        'Pass --source_path explicitly.'
    )


def _resolve_min_max_values_path(vqvae_config: dict) -> str:
    dataset_cfg = vqvae_config.get('dataset') or {}
    min_max_path = dataset_cfg.get('min_max_values_path')
    if not min_max_path:
        raise ValueError('dataset.min_max_values_path missing from VQ-VAE config.')
    return str(min_max_path)


def _resolve_level_target_time_frames(vqvae_config: dict, level: str) -> int:
    model_cfg = vqvae_config.get('model') or {}
    profile = ((model_cfg.get('level_profiles') or {}).get(level) or {})
    if 'target_time_frames' in profile:
        return int(profile['target_time_frames'])
    dataset_cfg = vqvae_config.get('dataset') or {}
    return int(dataset_cfg.get('target_time_frames', 2048))


def _select_windowed_entries(
    quantized_root: str,
    level: str,
    num_samples: int,
    seed: Optional[int],
    explicit_file: Optional[str] = None,
) -> List[Dict[str, object]]:
    if explicit_file:
        payload = torch.load(explicit_file, map_location='cpu', weights_only=False)
        if payload.get('format') != 'windowed_v1':
            raise ValueError(f'{explicit_file} is not a windowed quantized sample.')
        return [{'file': os.path.basename(explicit_file)}]

    manifest_path = os.path.join(quantized_root, WINDOWED_MANIFEST)
    if not os.path.isfile(manifest_path):
        return []

    entries = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if level in entry.get('eligible_levels', []):
                entries.append(entry)

    if not entries:
        return []

    rng = np.random.default_rng(seed)
    count = min(num_samples, len(entries))
    chosen_idx = rng.choice(len(entries), size=count, replace=False)
    return [entries[int(i)] for i in chosen_idx]


def _select_legacy_files(
    quantized_root: str,
    num_samples: int,
    seed: Optional[int],
    explicit_file: Optional[str] = None,
) -> List[str]:
    if explicit_file:
        return [explicit_file]

    files = sorted(
        os.path.join(quantized_root, name)
        for name in os.listdir(quantized_root)
        if name.endswith('_full_quantized.pt')
    )
    if not files:
        raise FileNotFoundError(f'No legacy _full_quantized.pt files found in {quantized_root}')

    rng = np.random.default_rng(seed)
    count = min(num_samples, len(files))
    chosen_idx = rng.choice(len(files), size=count, replace=False)
    return [files[int(i)] for i in chosen_idx]


def _decode_indices(vqvae, indices_2d: torch.Tensor) -> np.ndarray:
    if indices_2d.ndim != 2:
        raise ValueError(f'Expected indices shape (time, freq), got {tuple(indices_2d.shape)}')

    idx = indices_2d.unsqueeze(0).long().to(next(vqvae.parameters()).device)
    idx = idx.transpose(1, 2).contiguous()  # (B, freq, time)

    with torch.no_grad():
        batch_size, freq_bins, time_steps = idx.shape
        idx_flat = idx.reshape(-1)
        emb_flat = vqvae.vq.embedding[idx_flat]
        emb = emb_flat.view(batch_size, freq_bins, time_steps, -1)
        z_q = emb.permute(0, 3, 1, 2).contiguous()
        x_hat = vqvae.decoder(z_q)
        if vqvae.activation_layer is not None:
            x_hat = vqvae.activation_layer(x_hat)

    return x_hat.detach().cpu().permute(0, 2, 3, 1).numpy()


def _load_original_window(source_file: str, start_frame: int, target_time_frames: int) -> np.ndarray:
    spec = np.load(source_file)
    if spec.ndim == 3 and spec.shape[-1] == 1:
        spec = spec[:, :, 0]

    window = spec[:, start_frame:start_frame + target_time_frames]
    if window.shape[1] < target_time_frames:
        pad_width = target_time_frames - window.shape[1]
        window = np.pad(window, ((0, 0), (0, pad_width)), mode='constant')
    return window[..., np.newaxis].astype(np.float32, copy=False)


def _save_comparison_plot(original: np.ndarray, decoded: np.ndarray, out_path: str, title: str) -> Dict[str, float]:
    orig = original[:, :, 0]
    recon = decoded[:, :, 0]
    diff = np.abs(orig - recon)

    mse = float(np.mean((orig - recon) ** 2))
    mae = float(np.mean(diff))
    psnr = float(10 * np.log10((np.max(orig) ** 2) / mse)) if mse > 0 and np.max(orig) > 0 else float('inf')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axes[0].imshow(orig, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title('Original')
    axes[0].set_xlabel('Time Frames')
    axes[0].set_ylabel('Frequency Bins')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(recon, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('Decoded From Quantized Codes')
    axes[1].set_xlabel('Time Frames')
    axes[1].set_ylabel('Frequency Bins')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(diff, origin='lower', aspect='auto', cmap='hot')
    axes[2].set_title('Absolute Difference')
    axes[2].set_xlabel('Time Frames')
    axes[2].set_ylabel('Frequency Bins')
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle(
        f'{title}\nMSE={mse:.6f} | MAE={mae:.6f} | PSNR={psnr:.2f} dB',
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {'mse': mse, 'mae': mae, 'psnr': psnr}


def _save_single_spec(spec: np.ndarray, out_path: str, title: str) -> None:
    img = spec[:, :, 0]
    plt.figure(figsize=(8, 4))
    plt.imshow(img, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _prepare_min_max(
    source_file: str,
    min_max_values: dict,
    source_path: str,
) -> dict:
    mm = find_min_max_for_path(source_file, min_max_values, source_path)
    if mm is None:
        return {'min': -80.0, 'max': 0.0}
    return mm


def _inspect_windowed_sample(
    file_path: str,
    level: str,
    vqvae,
    target_time_frames: int,
    source_path: str,
    min_max_values: dict,
    sound_generator: SoundGenerator,
    audio_method: str,
    save_root: str,
) -> None:
    payload = torch.load(file_path, map_location='cpu', weights_only=False)
    source_basename = str(payload['source_basename'])
    start_frame = int(payload['start_frame'])
    total_frames = int(payload['total_frames'])
    indices = payload[level].long()

    source_file = os.path.join(source_path, source_basename)
    original_spec = _load_original_window(source_file, start_frame, target_time_frames)
    decoded_spec = _decode_indices(vqvae, indices)[0]
    decoded_spec = decoded_spec.astype(np.float32, copy=False)

    sample_dir = os.path.join(save_root, f'{source_basename}__start_{start_frame:08d}__{level}')
    os.makedirs(sample_dir, exist_ok=True)
    audio_dir = os.path.join(sample_dir, 'audio')
    spec_dir = os.path.join(sample_dir, 'spectrograms')
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(spec_dir, exist_ok=True)

    np.save(os.path.join(sample_dir, 'token_indices.npy'), indices.numpy())
    np.save(os.path.join(sample_dir, 'original_spec.npy'), original_spec)
    np.save(os.path.join(sample_dir, 'decoded_spec.npy'), decoded_spec)
    _save_single_spec(
        original_spec,
        os.path.join(spec_dir, 'original.png'),
        title=f'Original | {source_basename} | start={start_frame}',
    )
    _save_single_spec(
        decoded_spec,
        os.path.join(spec_dir, 'decoded.png'),
        title=f'Decoded | {source_basename} | start={start_frame}',
    )

    metrics = _save_comparison_plot(
        original_spec,
        decoded_spec,
        os.path.join(spec_dir, 'comparison.png'),
        title=f'{source_basename} | level={level} | start_frame={start_frame}',
    )

    mm = _prepare_min_max(source_file, min_max_values, source_path)
    original_audio = sound_generator.convert_spectrograms_to_audio(
        np.expand_dims(original_spec, axis=0), [mm], method=audio_method
    )[0]
    decoded_audio = sound_generator.convert_spectrograms_to_audio(
        np.expand_dims(decoded_spec, axis=0), [mm], method=audio_method
    )[0]

    sf.write(os.path.join(audio_dir, 'original.wav'), original_audio, sound_generator.sample_rate)
    sf.write(os.path.join(audio_dir, 'decoded.wav'), decoded_audio, sound_generator.sample_rate)

    metadata = {
        'format': payload.get('format', 'windowed_v1'),
        'level': level,
        'quantized_file': file_path,
        'source_file': source_file,
        'source_basename': source_basename,
        'start_frame': start_frame,
        'total_frames': total_frames,
        'timing': payload.get('timing').tolist() if torch.is_tensor(payload.get('timing')) else payload.get('timing'),
        'eligible_levels': payload.get('eligible_levels'),
        'metrics': metrics,
        'token_shape': list(indices.shape),
        'target_time_frames': target_time_frames,
    }
    with open(os.path.join(sample_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def _inspect_legacy_sample(
    file_path: str,
    level: str,
    vqvae,
    target_time_frames: int,
    source_path: str,
    min_max_values: dict,
    sound_generator: SoundGenerator,
    audio_method: str,
    save_root: str,
    start_frame_override: Optional[int],
    rng: np.random.Generator,
) -> None:
    payload = torch.load(file_path, map_location='cpu', weights_only=False)
    source_basename = _legacy_source_basename(file_path)
    source_file = os.path.join(source_path, source_basename)

    level_tokens = torch.as_tensor(payload[level], dtype=torch.long)
    total_frames = int(payload['total_frames'])
    time_steps_total = int(level_tokens.shape[0])
    ratio = max(1, total_frames // time_steps_total)
    window_token_steps = max(1, target_time_frames // ratio)

    max_start_frame = max(0, total_frames - target_time_frames)
    if start_frame_override is None:
        requested_start_frame = int(rng.integers(0, max_start_frame + 1)) if max_start_frame > 0 else 0
    else:
        requested_start_frame = max(0, min(int(start_frame_override), max_start_frame))

    token_start = min(level_tokens.shape[0] - window_token_steps, requested_start_frame // ratio)
    token_start = max(0, int(token_start))
    actual_start_frame = token_start * ratio

    token_slice = level_tokens[token_start:token_start + window_token_steps]
    if token_slice.shape[0] < window_token_steps:
        pad = torch.zeros((window_token_steps - token_slice.shape[0], token_slice.shape[1]), dtype=torch.long)
        token_slice = torch.cat([token_slice, pad], dim=0)

    original_spec = _load_original_window(source_file, actual_start_frame, target_time_frames)
    decoded_spec = _decode_indices(vqvae, token_slice)[0]
    decoded_spec = decoded_spec.astype(np.float32, copy=False)

    sample_dir = os.path.join(save_root, f'{source_basename}__start_{actual_start_frame:08d}__{level}')
    os.makedirs(sample_dir, exist_ok=True)
    audio_dir = os.path.join(sample_dir, 'audio')
    spec_dir = os.path.join(sample_dir, 'spectrograms')
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(spec_dir, exist_ok=True)

    np.save(os.path.join(sample_dir, 'token_indices.npy'), token_slice.numpy())
    np.save(os.path.join(sample_dir, 'original_spec.npy'), original_spec)
    np.save(os.path.join(sample_dir, 'decoded_spec.npy'), decoded_spec)
    _save_single_spec(
        original_spec,
        os.path.join(spec_dir, 'original.png'),
        title=f'Original | {source_basename} | start={actual_start_frame}',
    )
    _save_single_spec(
        decoded_spec,
        os.path.join(spec_dir, 'decoded.png'),
        title=f'Decoded | {source_basename} | start={actual_start_frame}',
    )

    metrics = _save_comparison_plot(
        original_spec,
        decoded_spec,
        os.path.join(spec_dir, 'comparison.png'),
        title=f'{source_basename} | level={level} | start_frame={actual_start_frame}',
    )

    mm = _prepare_min_max(source_file, min_max_values, source_path)
    original_audio = sound_generator.convert_spectrograms_to_audio(
        np.expand_dims(original_spec, axis=0), [mm], method=audio_method
    )[0]
    decoded_audio = sound_generator.convert_spectrograms_to_audio(
        np.expand_dims(decoded_spec, axis=0), [mm], method=audio_method
    )[0]

    sf.write(os.path.join(audio_dir, 'original.wav'), original_audio, sound_generator.sample_rate)
    sf.write(os.path.join(audio_dir, 'decoded.wav'), decoded_audio, sound_generator.sample_rate)

    metadata = {
        'format': 'legacy_full_song',
        'level': level,
        'quantized_file': file_path,
        'source_file': source_file,
        'source_basename': source_basename,
        'requested_start_frame': requested_start_frame,
        'actual_start_frame': actual_start_frame,
        'total_frames': total_frames,
        'ratio': ratio,
        'metrics': metrics,
        'token_shape': list(token_slice.shape),
        'target_time_frames': target_time_frames,
    }
    with open(os.path.join(sample_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description='Decode and inspect precomputed quantized datasets.')
    parser.add_argument('--quantized_path', type=str, required=True, help='Path to a quantized directory or a specific .pt file')
    parser.add_argument('--level', type=str, default='top', choices=['top', 'middle', 'bottom'], help='Which quantized level to inspect')
    parser.add_argument('--vqvae_model_dir', type=str, default=None, help='Override VQ-VAE run directory for the selected level')
    parser.add_argument('--transformer_config', type=str, default=None, help='Optional transformer prior config used to resolve VQ-VAE dirs')
    parser.add_argument('--source_path', type=str, default=None, help='Override processed spectrogram directory')
    parser.add_argument('--weights_file', type=str, default='best_model.pth', help='Checkpoint filename for the selected VQ-VAE')
    parser.add_argument('--samples', type=int, default=4, help='How many quantized examples to inspect')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--start_frame', type=int, default=None, help='Legacy full-song mode only: requested raw-frame start to inspect')
    parser.add_argument('--audio_method', type=str, default='griffinlim', choices=['griffinlim', 'istft'], help='Audio inversion method')
    parser.add_argument('--save_root', type=str, default='samples/quantized_dataset_inspection', help='Root directory for outputs')
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError(f'--samples must be > 0, got {args.samples}')

    _set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    explicit_file = args.quantized_path if os.path.isfile(args.quantized_path) else None
    quantized_root = os.path.dirname(args.quantized_path) if explicit_file else args.quantized_path
    if not os.path.isdir(quantized_root):
        raise FileNotFoundError(f'Quantized directory does not exist: {quantized_root}')

    vqvae_dir = _resolve_vqvae_dir(
        quantized_root=quantized_root,
        level=args.level,
        explicit_vqvae_dir=args.vqvae_model_dir,
        transformer_config=args.transformer_config,
    )
    vqvae = load_jukebox_model(vqvae_dir, args.level, device, args.weights_file)
    vqvae_config = load_config(os.path.join(vqvae_dir, 'config.yaml'))

    source_path = _resolve_source_path(
        quantized_root=quantized_root,
        explicit_source_path=args.source_path,
        vqvae_config=vqvae_config,
    )
    min_max_values_path = _resolve_min_max_values_path(vqvae_config)
    target_time_frames = _resolve_level_target_time_frames(vqvae_config, args.level)

    if not os.path.exists(min_max_values_path):
        raise FileNotFoundError(f'min_max_values.pkl not found at {min_max_values_path}')
    with open(min_max_values_path, 'rb') as f:
        min_max_values = pickle.load(f)

    dataset_cfg = vqvae_config.get('dataset') or {}
    sample_rate = int(dataset_cfg.get('sample_rate', 22050))
    hop_length = int(dataset_cfg.get('hop_length', 256))
    frame_size = int(dataset_cfg.get('frame_size', 2048))
    spectrogram_type_cfg = dataset_cfg.get('spectrogram_type')
    spectrogram_type = str(spectrogram_type_cfg).strip().lower() if spectrogram_type_cfg else (
        'mel' if 'mel' in str(source_path).lower() else 'linear'
    )
    n_mels = int(dataset_cfg.get('n_mels', 256))

    sound_generator = SoundGenerator(
        vqvae,
        hop_length=hop_length,
        sample_rate=sample_rate,
        n_fft=frame_size,
        spectrogram_type=spectrogram_type,
        n_mels=n_mels,
    )

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_root = os.path.join(args.save_root, f'{args.level}_{timestamp}')
    os.makedirs(save_root, exist_ok=True)

    windowed_entries = _select_windowed_entries(
        quantized_root=quantized_root,
        level=args.level,
        num_samples=args.samples,
        seed=args.seed,
        explicit_file=explicit_file,
    )

    summary = {
        'level': args.level,
        'quantized_root': quantized_root,
        'vqvae_dir': vqvae_dir,
        'source_path': source_path,
        'target_time_frames': target_time_frames,
        'audio_method': args.audio_method,
        'mode': 'windowed' if windowed_entries else 'legacy_full_song',
        'samples': [],
    }

    if windowed_entries:
        for entry in windowed_entries:
            file_path = explicit_file or os.path.join(quantized_root, entry['file'])
            _inspect_windowed_sample(
                file_path=file_path,
                level=args.level,
                vqvae=vqvae,
                target_time_frames=target_time_frames,
                source_path=source_path,
                min_max_values=min_max_values,
                sound_generator=sound_generator,
                audio_method=args.audio_method,
                save_root=save_root,
            )
            summary['samples'].append(file_path)
    else:
        legacy_files = _select_legacy_files(
            quantized_root=quantized_root,
            num_samples=args.samples,
            seed=args.seed,
            explicit_file=explicit_file,
        )
        for file_path in legacy_files:
            _inspect_legacy_sample(
                file_path=file_path,
                level=args.level,
                vqvae=vqvae,
                target_time_frames=target_time_frames,
                source_path=source_path,
                min_max_values=min_max_values,
                sound_generator=sound_generator,
                audio_method=args.audio_method,
                save_root=save_root,
                start_frame_override=args.start_frame,
                rng=rng,
            )
            summary['samples'].append(file_path)

    with open(os.path.join(save_root, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f'Saved quantized dataset inspection outputs to: {save_root}')


if __name__ == '__main__':
    main()
