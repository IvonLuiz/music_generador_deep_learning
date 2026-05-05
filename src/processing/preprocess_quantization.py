import argparse
import json
import os
import sys
from glob import glob
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_scripts.jukebox_utils import load_jukebox_model
from windowed_data_utils import (
    build_level_starts,
    build_timing_tensor,
    extract_window,
    source_stem,
)


WINDOWED_MANIFEST = 'windowed_manifest.jsonl'
DEFAULT_WINDOW_OVERLAP_FRACTION = 0.50


def _step_from_overlap(window_size: int, overlap_fraction: float, level_name: str) -> int:
    """!
    @brief Calculate the step size from the window size and overlap fraction.
    @param window_size The size of the window in frames for the level.
    @param overlap_fraction The desired overlap fraction between windows (e.g., 0.75 for 75% overlap).
    @param level_name The name of the level (e.g., 'top', 'middle', 'bottom') for error messages.
    @return The calculated step size in frames, ensuring it is at least 1 and does not exceed the window size.
    """
    
    if window_size <= 0:
        raise ValueError(f'{level_name}_time_frames must be > 0, got {window_size}')
    if not 0.0 <= overlap_fraction < 1.0:
        raise ValueError(f'overlap_fraction must be in [0, 1), got {overlap_fraction}')

    step = int(round(window_size * (1.0 - overlap_fraction)))
    return max(1, min(window_size, step))


def _resolve_step_frames(
    explicit_step: Optional[int],
    window_size: int,
    overlap_fraction: float,
    level_name: str,
) -> int:
    """!
    @brief Resolve the step frames for a given level, using either an explicit value or deriving from the overlap fraction.
    @param explicit_step An optional explicit step frame count. If provided, this value is used directly after validation.
    @param window_size The size of the window in frames for the level.
    @param overlap_fraction The desired overlap fraction between windows, used to derive the step if explicit_step is not provided.
    @param level_name The name of the level (e.g., 'top', 'middle', 'bottom') for error messages.
    @return The validated step size in frames for the requested level.
    """
    if explicit_step is not None:
        if explicit_step <= 0:
            raise ValueError(f'{level_name}_step_frames must be > 0, got {explicit_step}')
        return int(explicit_step)

    return _step_from_overlap(window_size, overlap_fraction, level_name)


def _encode_level_windows(model, batch_windows: np.ndarray, device: torch.device) -> torch.Tensor:
    """!
    @brief Encode a batch of spectrogram windows into VQ code indices.
    @param model Loaded Jukebox VQ-VAE model for one hierarchy level.
    @param batch_windows NumPy batch shaped (B, Freq, Time) containing normalized spectrogram windows.
    @param device Torch device used for the VQ-VAE forward pass.
    @return Integer tensor shaped (B, Time_latent, Freq_latent) on CPU for serialization.
    """
    x_batch = torch.from_numpy(batch_windows).unsqueeze(1).float().to(device)
    indices = model.encode_to_indices(x_batch)
    return indices.transpose(1, 2).contiguous().cpu().to(torch.int32)


def _build_anchor_schedule(
    total_frames: int,
    top_time_frames: int,
    middle_time_frames: int,
    bottom_time_frames: int,
    top_step_frames: int,
    middle_step_frames: int,
    bottom_step_frames: int,
) -> Dict[int, List[str]]:
    """!
    @brief Build a schedule of anchor start frames mapped to their eligible 
    levels based on the provided window and step configurations.
    @param total_frames The total number of frames in the spectrogram.
    @param top_time_frames The window size in frames for the top level.
    @param middle_time_frames The window size in frames for the middle level.
    @param bottom_time_frames The window size in frames for the bottom level.
    @param top_step_frames The step size in frames for the top level.
    @param middle_step_frames The step size in frames for the middle level.
    @param bottom_step_frames The step size in frames for the bottom level.
    @return A dictionary mapping each anchor start frame (int) to a list of 
    eligible level names (List[str]) that can be trained at that anchor based
    on the window and step configurations.
    """
    top_starts = build_level_starts(total_frames, top_time_frames, top_step_frames)
    middle_starts = build_level_starts(total_frames, middle_time_frames, middle_step_frames)
    bottom_starts = build_level_starts(total_frames, bottom_time_frames, bottom_step_frames)

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
    """!
    @brief Quantize whole songs chunk-by-chunk and save one stitched latent file per source spectrogram.
    @param vqvae_dirs Mapping with the trained VQ-VAE directory for the top, middle, and bottom levels.
    @param source_npy_path Directory containing source spectrogram `.npy` files.
    @param output_dir Directory where stitched quantized `.pt` files will be saved.
    @param chunk_size Raw spectrogram time frames processed at once to stay within GPU memory limits.
    @param device Torch device used for quantization.
    @param weights_file Checkpoint filename to load from each VQ-VAE directory.
    @return None. Quantized latent files are written to `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)
    models = {lvl: load_jukebox_model(vqvae_dirs[lvl], lvl, device, weights_file).eval() 
              for lvl in ['top', 'middle', 'bottom']}
    
    files = sorted(glob(os.path.join(source_npy_path, "*.npy")))

    with torch.no_grad():
        for f in tqdm(files, desc="Quantizing in Chunks"):
            spec = np.load(f)  # [Freq, Total_Time]
            total_frames = spec.shape[1]
            
            # Collect latent index chunks on CPU before stitching the full-song timeline.
            all_indices = {'top': [], 'middle': [], 'bottom': []}

            for start in range(0, total_frames, chunk_size):
                end = min(start + chunk_size, total_frames)
                
                # Pad the last chunk temporarily so the deepest top encoder can downsample cleanly.
                chunk = spec[:, start:end]
                actual_len = chunk.shape[1]
                
                pad_val = 0
                if actual_len % 32 != 0:
                    pad_val = 32 - (actual_len % 32)
                    chunk = np.pad(chunk, ((0, 0), (0, pad_val)), mode='constant')

                x_chunk = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).float().to(device)

                for lvl in ['top', 'middle', 'bottom']:
                    m = models[lvl]
                    idx = m.encode_to_indices(x_chunk)
                    
                    # Convert to [Time, Freq] and drop latent columns introduced only by padding.
                    ratio = 32 if lvl == 'top' else 16 if lvl == 'middle' else 8
                    tokens_to_keep = actual_len // ratio
                    
                    idx_np = idx.squeeze(0).transpose(0, 1).cpu().numpy()
                    all_indices[lvl].append(idx_np[:tokens_to_keep, :])

            # Stitch the per-chunk latent timelines on CPU once the whole song has been encoded.
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
            
            if device.type == 'cuda':
                torch.cuda.empty_cache()


def precompute_windowed_examples(
    vqvae_dirs,
    source_npy_path,
    output_dir,
    top_time_frames=2048,
    middle_time_frames=512,
    bottom_time_frames=128,
    top_step_frames=None,
    middle_step_frames=None,
    bottom_step_frames=None,
    overlap_fraction=DEFAULT_WINDOW_OVERLAP_FRACTION,
    batch_size=8,
    sample_rate=22050,
    hop_length=256,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    weights_file='best_model.pth',
):
    """!
    @brief Quantize aligned top, middle, and bottom windows into seam-free training examples for transformer priors.
    @param vqvae_dirs Mapping with the trained VQ-VAE directory for the top, middle, and bottom levels.
    @param source_npy_path Directory containing source spectrogram `.npy` files.
    @param output_dir Directory where windowed quantized `.pt` files and metadata will be saved.
    @param top_time_frames Raw spectrogram window size for the top prior examples.
    @param middle_time_frames Raw spectrogram window size for the middle prior examples.
    @param bottom_time_frames Raw spectrogram window size for the bottom prior examples.
    @param top_step_frames Optional explicit raw-frame hop for top windows. If omitted, it is derived from `overlap_fraction`.
    @param middle_step_frames Optional explicit raw-frame hop for middle windows. If omitted, it is derived from `overlap_fraction`.
    @param bottom_step_frames Optional explicit raw-frame hop for bottom windows. If omitted, it is derived from `overlap_fraction`.
    @param overlap_fraction Requested window overlap fraction used to derive step sizes when explicit hops are not provided.
    @param batch_size Number of anchor positions quantized together per forward pass.
    @param sample_rate Audio sample rate used to convert start frames into seconds.
    @param hop_length Spectrogram hop length used to convert start frames into seconds.
    @param device Torch device used for quantization.
    @param weights_file Checkpoint filename to load from each VQ-VAE directory.
    @return None. Quantized examples plus manifest/config metadata are written to `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)
    top_step_frames = _resolve_step_frames(top_step_frames, top_time_frames, overlap_fraction, 'top')
    middle_step_frames = _resolve_step_frames(middle_step_frames, middle_time_frames, overlap_fraction, 'middle')
    bottom_step_frames = _resolve_step_frames(bottom_step_frames, bottom_time_frames, overlap_fraction, 'bottom')
    effective_overlap = {
        'top': 1.0 - (top_step_frames / float(top_time_frames)),
        'middle': 1.0 - (middle_step_frames / float(middle_time_frames)),
        'bottom': 1.0 - (bottom_step_frames / float(bottom_time_frames)),
    }

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
        'requested_overlap_fraction': float(overlap_fraction),
        'effective_overlap_fraction': effective_overlap,
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
            source_file_stem = source_stem(file_path)
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
                        [extract_window(spec, start_frame=s, window_size=top_time_frames) for s in batch_starts],
                        axis=0,
                    )
                    middle_batch = np.stack(
                        [extract_window(spec, start_frame=s, window_size=middle_time_frames) for s in batch_starts],
                        axis=0,
                    )
                    bottom_batch = np.stack(
                        [extract_window(spec, start_frame=s, window_size=bottom_time_frames) for s in batch_starts],
                        axis=0,
                    )

                    top_indices = _encode_level_windows(models['top'], top_batch, device)
                    middle_indices = _encode_level_windows(models['middle'], middle_batch, device)
                    bottom_indices = _encode_level_windows(models['bottom'], bottom_batch, device)

                    for batch_idx, start_frame in enumerate(batch_starts):
                        eligible_levels = anchor_to_levels[int(start_frame)]
                        filename = f'{source_file_stem}__start_{int(start_frame):08d}_window_quantized.pt'
                        payload = {
                            'format': 'windowed_v1',
                            'source_basename': source_basename,
                            'source_stem': source_file_stem,
                            'start_frame': int(start_frame),
                            'total_frames': total_frames,
                            'timing': build_timing_tensor(
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
                                    'source_stem': source_file_stem,
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
    """!
    @brief Parse CLI arguments and run quantization preprocessing in either windowed or legacy full-song mode.
    @param None
    @return None. The selected quantized dataset is written to disk and progress is printed to stdout.
    """
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
    parser.add_argument(
        '--overlap_fraction',
        type=float,
        default=DEFAULT_WINDOW_OVERLAP_FRACTION,
        help='Default overlap used to derive step frames when explicit step args are omitted (default: 0.50).',
    )
    parser.add_argument(
        '--top_step_frames',
        type=int,
        default=None,
        help='Top-level anchor step. Omit to derive from --overlap_fraction; with defaults this is 1024.',
    )
    parser.add_argument(
        '--middle_step_frames',
        type=int,
        default=None,
        help='Middle-level anchor step. Omit to derive from --overlap_fraction; with defaults this is 256.',
    )
    parser.add_argument(
        '--bottom_step_frames',
        type=int,
        default=None,
        help='Bottom-level anchor step. Omit to derive from --overlap_fraction; with defaults this is 64.',
    )
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
    top_step_frames = _resolve_step_frames(
        args.top_step_frames, args.top_time_frames, args.overlap_fraction, 'top'
    )
    middle_step_frames = _resolve_step_frames(
        args.middle_step_frames, args.middle_time_frames, args.overlap_fraction, 'middle'
    )
    bottom_step_frames = _resolve_step_frames(
        args.bottom_step_frames, args.bottom_time_frames, args.overlap_fraction, 'bottom'
    )
    effective_overlap = {
        'top': 1.0 - (top_step_frames / float(args.top_time_frames)),
        'middle': 1.0 - (middle_step_frames / float(args.middle_time_frames)),
        'bottom': 1.0 - (bottom_step_frames / float(args.bottom_time_frames)),
    }
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
        f'bottom={args.bottom_time_frames}, top_step={top_step_frames}, '
        f'middle_step={middle_step_frames}, bottom_step={bottom_step_frames}, '
        f'sr={args.sample_rate}, hop={args.hop_length}'
    )
    print(
        'Window overlap: '
        f"requested={args.overlap_fraction:.3f}, "
        f"effective top={effective_overlap['top']:.3f}, "
        f"middle={effective_overlap['middle']:.3f}, "
        f"bottom={effective_overlap['bottom']:.3f}"
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
            top_step_frames=top_step_frames,
            middle_step_frames=middle_step_frames,
            bottom_step_frames=bottom_step_frames,
            overlap_fraction=args.overlap_fraction,
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
