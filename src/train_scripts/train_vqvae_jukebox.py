import argparse
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.runners import run_jukebox_vqvae_training


def main():
    parser = argparse.ArgumentParser(description='Train a Jukebox-style VQ-VAE level.')
    parser.add_argument('--config', type=str, default='./config/config_jukebox.yaml')
    parser.add_argument(
        '--level',
        type=str,
        choices=['bottom', 'middle', 'top'],
        default=None,
        help='Override config model.selected_level.',
    )
    parser.add_argument(
        '--resume-checkpoint',
        '--resume',
        dest='resume_checkpoint',
        type=str,
        default=None,
        help='Resume from a previous latest_model.pth/best_model.pth checkpoint.',
    )
    args = parser.parse_args()
    run_dir = run_jukebox_vqvae_training(
        config_path=args.config,
        level_override=args.level,
        resume_checkpoint=args.resume_checkpoint,
    )
    print('Jukebox VQ-VAE training complete. Artifacts saved to:', run_dir)


if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    main()
