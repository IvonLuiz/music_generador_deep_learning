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

