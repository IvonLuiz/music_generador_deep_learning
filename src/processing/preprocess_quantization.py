import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm
from glob import glob

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_scripts.jukebox_utils import load_jukebox_model, extract_song_prefix

def precompute_and_save_indices(
    vqvae_dirs: dict,
    source_npy_path: str,
    output_dir: str,
    device: torch.device,
    top_time_frames: int,
    middle_time_frames: int,
    bottom_time_frames: int,
    sample_rate: int,
    hop_length: int,
    segment_overlap: float,
    weights_file: str,
):
    os.makedirs(output_dir, exist_ok=True)

    top_tf = int(top_time_frames)
    mid_tf = int(middle_time_frames)
    bot_tf = int(bottom_time_frames)
    if top_tf <= 0 or mid_tf <= 0 or bot_tf <= 0:
        raise ValueError('All time-frame values must be positive integers.')
    if top_tf % mid_tf != 0 or top_tf % bot_tf != 0:
        raise ValueError(
            f'Incompatible level windows: top={top_tf}, middle={mid_tf}, bottom={bot_tf}. '
            'top_time_frames must be divisible by middle_time_frames and bottom_time_frames.'
        )
    if not (0.0 <= float(segment_overlap) < 1.0):
        raise ValueError(f'segment_overlap must be in [0, 1), got {segment_overlap}.')

    stride_mid = top_tf // mid_tf
    stride_bot = top_tf // bot_tf

    # Same temporal logic used in preprocess_audio.py.
    segment_samples = max(1, (top_tf - 1) * int(hop_length))
    segment_duration_s = segment_samples / float(sample_rate)
    hop_samples = max(1, int(segment_samples * (1.0 - float(segment_overlap))))
    hop_duration_s = hop_samples / float(sample_rate)
    slice_duration_s = segment_duration_s / float(stride_bot)
    
    # Load Models
    models = {
        lvl: load_jukebox_model(vqvae_dirs[lvl], lvl, device, weights_file).eval()
        for lvl in ['top', 'middle', 'bottom']
    }

    files = sorted(glob(os.path.join(source_npy_path, "**/*.npy"), recursive=True))
    if len(files) == 0:
        raise ValueError(f'No .npy files found under: {source_npy_path}')
    
    # Discovery pass for timing (standard Jukebox global duration logic)
    song_max_idx = {}
    for f in files:
        prefix = extract_song_prefix(f)
        idx = int(f.split("_segment_")[-1].split(".")[0])
        song_max_idx[prefix] = max(song_max_idx.get(prefix, 0), idx)

    # Calculation Pass
    with torch.no_grad():
        pbar = tqdm(total=len(files), desc="Quantizing Dataset")
        for f in files:
            prefix = extract_song_prefix(f)
            seg_idx = int(f.split("_segment_")[-1].split(".")[0])

            # Load single spectrogram and ensure top window
            spec = np.load(f)
            if spec.shape[1] < top_tf:
                spec = np.pad(spec, ((0, 0), (0, top_tf - spec.shape[1])))
            spec = spec[:, :top_tf]

            x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float().to(device)

            # Quantize at all levels
            latents = {}
            for lvl in ['top', 'middle', 'bottom']:
                m = models[lvl]
                z = m.encoder(x)
                z = m.pre_vq_conv(z)
                _, idx, _, _, _ = m.vq(z)
                latents[lvl] = idx.squeeze(0).transpose(0, 1).cpu().numpy()  # (Time, Freq)

            middle_len = int(latents['middle'].shape[0])
            bottom_len = int(latents['bottom'].shape[0])
            middle_slice_len = middle_len // stride_mid
            bottom_slice_len = bottom_len // stride_bot

            samples = []
            for j in range(stride_bot):
                container_start_s = seg_idx * hop_duration_s
                slice_start_s = container_start_s + (j * slice_duration_s)
                total_song_s = (song_max_idx[prefix] * hop_duration_s) + segment_duration_s
                fraction = np.clip(slice_start_s / max(total_song_s, 1e-8), 0.0, 1.0)

                middle_slice_idx = j // stride_mid
                middle_start = middle_slice_idx * middle_slice_len
                middle_end = (middle_slice_idx + 1) * middle_slice_len
                bottom_start = j * bottom_slice_len
                bottom_end = (j + 1) * bottom_slice_len

                sample = {
                    "top": latents['top'],
                    "middle": latents['middle'][middle_start:middle_end],
                    "bottom": latents['bottom'][bottom_start:bottom_end],
                    "timing": np.array([slice_start_s, total_song_s, fraction], dtype=np.float32),
                }
                samples.append(sample)

            out_name = os.path.basename(f).replace(".npy", "_quantized.pt")
            torch.save(samples, os.path.join(output_dir, out_name))
            pbar.update(1)
        pbar.close()


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
    parser.add_argument('--sample_rate', type=int, default=22050, help='Sample rate used in preprocess_audio')
    parser.add_argument('--hop_length', type=int, default=256, help='Hop length used in preprocess_audio')
    parser.add_argument('--segment_overlap', type=float, default=0.5, help='Segment overlap used in preprocess_audio')
    
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
        f'bottom={args.bottom_time_frames}, sr={args.sample_rate}, hop={args.hop_length}, '
        f'overlap={args.segment_overlap}'
    )
    print()

    precompute_and_save_indices(
        vqvae_dirs=vqvae_dirs,
        source_npy_path=args.source_path,
        output_dir=args.output_dir,
        device=device,
        top_time_frames=args.top_time_frames,
        middle_time_frames=args.middle_time_frames,
        bottom_time_frames=args.bottom_time_frames,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        segment_overlap=args.segment_overlap,
        weights_file=args.weights_file,
    )
    
    print(f'\n✓ Quantization preprocessing complete!')
    print(f'Quantized files saved to: {args.output_dir}')


if __name__ == '__main__':
    main()

