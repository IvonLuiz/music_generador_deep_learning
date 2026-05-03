import argparse
import json
import os
import sys
from glob import glob
from typing import Dict, Iterable, List

import numpy as np
import torch
from tqdm import tqdm

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_scripts.jukebox_utils import load_jukebox_model


WINDOWED_MANIFEST = 'windowed_manifest.jsonl'


def _source_stem(file_path: str) -> str:
    return os.path.basename(file_path).replace('.npy', '')


def _build_level_starts(total_frames: int, window_size: int, step: int) -> List[int]:
    if window_size <= 0:
        raise ValueError(f'window_size must be > 0, got {window_size}')
    if step <= 0:
        raise ValueError(f'step must be > 0, got {step}')
    if total_frames <= window_size:
        return [0]

    starts = list(range(0, total_frames - window_size + 1, step))
    last_start = total_frames - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _extract_window(spec: np.ndarray, start_frame: int, window_size: int) -> np.ndarray:
    chunk = spec[:, start_frame:start_frame + window_size]
    if chunk.shape[1] < window_size:
        pad_width = window_size - chunk.shape[1]
        chunk = np.pad(chunk, ((0, 0), (0, pad_width)), mode='constant')
    return chunk.astype(np.float32, copy=False)


def _encode_level_windows(model, batch_windows: np.ndarray, device: torch.device) -> torch.Tensor:
    x_batch = torch.from_numpy(batch_windows).unsqueeze(1).float().to(device)
    encoded = model.encoder(x_batch)
    pre_vq = model.pre_vq_conv(encoded)
    _, indices, _, _, _ = model.vq(pre_vq)
    return indices.transpose(1, 2).contiguous().cpu().to(torch.int32)


def _build_timing(start_frame: int, total_frames: int, sample_rate: int, hop_length: int) -> torch.Tensor:
    start_time_s = (start_frame * hop_length) / sample_rate
    total_duration_s = (total_frames * hop_length) / sample_rate
    fraction = start_time_s / max(total_duration_s, 1e-6)
    return torch.tensor([start_time_s, total_duration_s, fraction], dtype=torch.float32)


def _build_anchor_schedule(
    total_frames: int,
    top_time_frames: int,
    middle_time_frames: int,
    bottom_time_frames: int,
    top_step_frames: int,
    middle_step_frames: int,
    bottom_step_frames: int,
) -> Dict[int, List[str]]:
    top_starts = _build_level_starts(total_frames, top_time_frames, top_step_frames)
    middle_starts = _build_level_starts(total_frames, middle_time_frames, middle_step_frames)
    bottom_starts = _build_level_starts(total_frames, bottom_time_frames, bottom_step_frames)

    anchor_to_levels: Dict[int, List[str]] = {}
    for level_name, starts in (
        ('top', top_starts),
        ('middle', middle_starts),
        ('bottom', bottom_starts),
    ):
        for start in starts:
            anchor_to_levels.setdefault(int(start), []).append(level_name)

    for levels in anchor_to_levels.values():
        levels.sort(key=lambda name: {'top': 0, 'middle': 1, 'bottom': 2}[name])
    return anchor_to_levels


def precompute_full_songs(
    vqvae_dirs,
    source_npy_path,
    output_dir,
    chunk_size=2048,  # Process in 2048-frame chunks to manage memory
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    weights_file="best_model.pth"
):
    os.makedirs(output_dir, exist_ok=True)
    models = {lvl: load_jukebox_model(vqvae_dirs[lvl], lvl, device, weights_file).eval() 
              for lvl in ['top', 'middle', 'bottom']}
    
    files = sorted(glob(os.path.join(source_npy_path, "*.npy")))

    with torch.no_grad():
        for f in tqdm(files, desc="Quantizing in Chunks"):
            spec = np.load(f)  # [Freq, Total_Time]
            total_frames = spec.shape[1]
            
            # Dicionários para guardar os pedaços de índices de cada nível
            all_indices = {'top': [], 'middle': [], 'bottom': []}

            # Loop por blocos temporais
            for start in range(0, total_frames, chunk_size):
                end = min(start + chunk_size, total_frames)
                
                # Extrai o pedaço e garante que é múltiplo de 32 para evitar erros de shape no encoder
                chunk = spec[:, start:end]
                actual_len = chunk.shape[1]
                
                # Se o último pedaço não for múltiplo de 32, fazemos um padding temporário
                pad_val = 0
                if actual_len % 32 != 0:
                    pad_val = 32 - (actual_len % 32)
                    chunk = np.pad(chunk, ((0, 0), (0, pad_val)), mode='constant')

                x_chunk = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).float().to(device)

                for lvl in ['top', 'middle', 'bottom']:
                    m = models[lvl]
                    # Forward pass do chunk
                    z = m.encoder(x_chunk)
                    z = m.pre_vq_conv(z)
                    _, idx, _, _, _ = m.vq(z)
                    
                    # Converte para [Time, Freq] e remove o padding se necessário
                    # O downsampling ratio muda por nível: Top=32, Mid=16, Bot=8
                    ratio = 32 if lvl == 'top' else 16 if lvl == 'middle' else 8
                    tokens_to_keep = actual_len // ratio
                    
                    # idx shape: [1, Freq, Time_steps]
                    idx_np = idx.squeeze(0).transpose(0, 1).cpu().numpy()
                    all_indices[lvl].append(idx_np[:tokens_to_keep, :])

            # Junta todos os chunks processados no CPU
            full_latents = {
                lvl: np.concatenate(all_indices[lvl], axis=0) 
                for lvl in ['top', 'middle', 'bottom']
            }

            save_data = {
                'top': full_latents['top'],
                'middle': full_latents['middle'],
                'bottom': full_latents['bottom'],
                'total_frames': total_frames
            }
            
            out_name = os.path.basename(f).replace(".npy", "_full_quantized.pt")
            torch.save(save_data, os.path.join(output_dir, out_name))
            
            # Limpa cache da GPU após cada música para garantir estabilidade
            torch.cuda.empty_cache()


def precompute_windowed_examples(
    vqvae_dirs,
    source_npy_path,
    output_dir,
    top_time_frames=2048,
    middle_time_frames=512,
    bottom_time_frames=128,
    top_step_frames=2048,
    middle_step_frames=512,
    bottom_step_frames=128,
    batch_size=8,
    sample_rate=22050,
    hop_length=256,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    weights_file='best_model.pth',
):
    os.makedirs(output_dir, exist_ok=True)
    models = {
        lvl: load_jukebox_model(vqvae_dirs[lvl], lvl, device, weights_file).eval()
        for lvl in ['top', 'middle', 'bottom']
    }

    files = sorted(glob(os.path.join(source_npy_path, '*.npy')))
    manifest_path = os.path.join(output_dir, WINDOWED_MANIFEST)
    config_path = os.path.join(output_dir, 'windowed_quantization_config.json')

    config_payload = {
        'format': 'windowed_v1',
        'source_path': source_npy_path,
        'top_time_frames': int(top_time_frames),
        'middle_time_frames': int(middle_time_frames),
        'bottom_time_frames': int(bottom_time_frames),
        'top_step_frames': int(top_step_frames),
        'middle_step_frames': int(middle_step_frames),
        'bottom_step_frames': int(bottom_step_frames),
        'batch_size': int(batch_size),
        'sample_rate': int(sample_rate),
        'hop_length': int(hop_length),
        'weights_file': weights_file,
        'vqvae_dirs': dict(vqvae_dirs),
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_payload, f, indent=2)

    total_examples = 0
    for file_path in files:
        total_frames = int(np.load(file_path, mmap_mode='r').shape[1])
        total_examples += len(
            _build_anchor_schedule(
                total_frames=total_frames,
                top_time_frames=top_time_frames,
                middle_time_frames=middle_time_frames,
                bottom_time_frames=bottom_time_frames,
                top_step_frames=top_step_frames,
                middle_step_frames=middle_step_frames,
                bottom_step_frames=bottom_step_frames,
            )
        )

    print(
        'Precomputing windowed hierarchical indices '
        f'(examples={total_examples}, top_step={top_step_frames}, '
        f'middle_step={middle_step_frames}, bottom_step={bottom_step_frames}, batch_size={batch_size})...'
    )

    with open(manifest_path, 'w', encoding='utf-8') as manifest_file, torch.no_grad():
        progress = tqdm(total=total_examples, desc='Quantizing Windowed Examples')
        current_batch_size = max(1, int(batch_size))

        for file_path in files:
            spec = np.load(file_path)
            total_frames = int(spec.shape[1])
            source_stem = _source_stem(file_path)
            source_basename = os.path.basename(file_path)

            anchor_to_levels = _build_anchor_schedule(
                total_frames=total_frames,
                top_time_frames=top_time_frames,
                middle_time_frames=middle_time_frames,
                bottom_time_frames=bottom_time_frames,
                top_step_frames=top_step_frames,
                middle_step_frames=middle_step_frames,
                bottom_step_frames=bottom_step_frames,
            )
            starts = sorted(anchor_to_levels.keys())

            i = 0
            while i < len(starts):
                end = min(i + current_batch_size, len(starts))
                batch_starts = starts[i:end]
                try:
                    top_batch = np.stack(
                        [_extract_window(spec, start_frame=s, window_size=top_time_frames) for s in batch_starts],
                        axis=0,
                    )
                    middle_batch = np.stack(
                        [_extract_window(spec, start_frame=s, window_size=middle_time_frames) for s in batch_starts],
                        axis=0,
                    )
                    bottom_batch = np.stack(
                        [_extract_window(spec, start_frame=s, window_size=bottom_time_frames) for s in batch_starts],
                        axis=0,
                    )

                    top_indices = _encode_level_windows(models['top'], top_batch, device)
                    middle_indices = _encode_level_windows(models['middle'], middle_batch, device)
                    bottom_indices = _encode_level_windows(models['bottom'], bottom_batch, device)

                    for batch_idx, start_frame in enumerate(batch_starts):
                        eligible_levels = anchor_to_levels[int(start_frame)]
                        filename = f'{source_stem}__start_{int(start_frame):08d}_window_quantized.pt'
                        payload = {
                            'format': 'windowed_v1',
                            'source_basename': source_basename,
                            'source_stem': source_stem,
                            'start_frame': int(start_frame),
                            'total_frames': total_frames,
                            'timing': _build_timing(
                                start_frame=int(start_frame),
                                total_frames=total_frames,
                                sample_rate=sample_rate,
                                hop_length=hop_length,
                            ),
                            'eligible_levels': list(eligible_levels),
                            'top': top_indices[batch_idx].clone(),
                            'middle': middle_indices[batch_idx].clone(),
                            'bottom': bottom_indices[batch_idx].clone(),
                        }
                        torch.save(payload, os.path.join(output_dir, filename))
                        manifest_file.write(
                            json.dumps(
                                {
                                    'file': filename,
                                    'source_basename': source_basename,
                                    'source_stem': source_stem,
                                    'start_frame': int(start_frame),
                                    'total_frames': total_frames,
                                    'eligible_levels': list(eligible_levels),
                                }
                            ) + '\n'
                        )

                    progress.update(end - i)
                    i = end

                except RuntimeError as exc:
                    if 'out of memory' in str(exc).lower() and device.type == 'cuda' and current_batch_size > 1:
                        new_batch_size = max(1, current_batch_size // 2)
                        print(
                            f'CUDA OOM while window-quantizing {source_basename} at batch_size={current_batch_size}. '
                            f'Retrying with batch_size={new_batch_size}.'
                        )
                        current_batch_size = new_batch_size
                        torch.cuda.empty_cache()
                        continue
                    progress.close()
                    raise

            if device.type == 'cuda':
                torch.cuda.empty_cache()

        progress.close()


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess and quantize spectrograms for hierarchical transformer prior training.'
    )
    
    parser.add_argument(
        '--source_path',
        type=str,
        default='./data/processed/maestro/',
        help='Path to directory containing .npy spectrogram files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/processed/maestro_quantized/',
        help='Output directory for quantized .pt files'
    )
    parser.add_argument(
        '--top_model_dir',
        type=str,
        default='./models/jukebox_vq_vae/jukebox_vqvae_maestro_top/2026-04-26_02-13-14',
        help='Path to trained top-level VQ-VAE model'
    )
    parser.add_argument(
        '--middle_model_dir',
        type=str,
        default='./models/jukebox_vq_vae/jukebox_vqvae_maestro_middle/2026-04-26_17-30-30',
        help='Path to trained middle-level VQ-VAE model'
    )
    parser.add_argument(
        '--bottom_model_dir',
        type=str,
        default='./models/jukebox_vq_vae/jukebox_vqvae_maestro_bottom/2026-04-26_22-26-06',
        help='Path to trained bottom-level VQ-VAE model'
    )
    parser.add_argument(
        '--weights_file',
        type=str,
        default='best_model.pth',
        help='Checkpoint filename for VQ-VAE models'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for computation (cuda or cpu)'
    )
    parser.add_argument('--top_time_frames', type=int, default=2048, help='Top-level window (same as preprocess_audio TARGET_TIME_FRAMES)')
    parser.add_argument('--middle_time_frames', type=int, default=512, help='Middle-level window')
    parser.add_argument('--bottom_time_frames', type=int, default=128, help='Bottom-level window')
    parser.add_argument('--top_step_frames', type=int, default=2048, help='Top-level anchor step. Default keeps one top example per 2048-frame span.')
    parser.add_argument('--middle_step_frames', type=int, default=512, help='Middle-level anchor step. Default keeps four middle examples per 2048-frame span.')
    parser.add_argument('--bottom_step_frames', type=int, default=128, help='Bottom-level anchor step. Default keeps sixteen bottom examples per 2048-frame span.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for windowed quantization.')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Sample rate used in preprocess_audio')
    parser.add_argument('--hop_length', type=int, default=256, help='Hop length used in preprocess_audio')
    parser.add_argument(
        '--mode',
        type=str,
        default='windowed',
        choices=['windowed', 'full_song_legacy'],
        help='Quantization format to generate. `windowed` is the seam-free training format.',
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.isdir(args.source_path):
        print(f"Error: Source path does not exist: {args.source_path}")
        sys.exit(1)
    
    for model_path in [args.top_model_dir, args.middle_model_dir, args.bottom_model_dir]:
        if not os.path.isdir(model_path):
            print(f"Error: Model path does not exist: {model_path}")
            sys.exit(1)
    
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    vqvae_dirs = {
        'top': args.top_model_dir,
        'middle': args.middle_model_dir,
        'bottom': args.bottom_model_dir,
    }
    
    print(f'Source path: {args.source_path}')
    print(f'Output directory: {args.output_dir}')
    print(f'Weights file: {args.weights_file}')
    print(
        f'Temporal config: top={args.top_time_frames}, middle={args.middle_time_frames}, '
        f'bottom={args.bottom_time_frames}, top_step={args.top_step_frames}, '
        f'middle_step={args.middle_step_frames}, bottom_step={args.bottom_step_frames}, '
        f'sr={args.sample_rate}, hop={args.hop_length}'
    )
    print()

    if args.mode == 'windowed':
        precompute_windowed_examples(
            vqvae_dirs=vqvae_dirs,
            source_npy_path=args.source_path,
            output_dir=args.output_dir,
            top_time_frames=args.top_time_frames,
            middle_time_frames=args.middle_time_frames,
            bottom_time_frames=args.bottom_time_frames,
            top_step_frames=args.top_step_frames,
            middle_step_frames=args.middle_step_frames,
            bottom_step_frames=args.bottom_step_frames,
            batch_size=args.batch_size,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            device=device,
            weights_file=args.weights_file,
        )
    else:
        precompute_full_songs(
            vqvae_dirs=vqvae_dirs,
            source_npy_path=args.source_path,
            output_dir=args.output_dir,
            device=device,
            weights_file=args.weights_file
        )
    
    print(f'\n✓ Quantization preprocessing complete!')
    print(f'Quantized files saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
