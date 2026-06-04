import argparse
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.runners import run_two_level_vqvae_training


def main():
    parser = argparse.ArgumentParser(description='Train a two-level hierarchical VQ-VAE.')
    parser.add_argument('--config', type=str, default='./config/config_vqvae_hierarchical.yaml')
    parser.add_argument(
        '--resume-checkpoint',
        type=str,
        default=None,
        help='Resume from a previous model.pth/best_model.pth checkpoint.',
    )
    args = parser.parse_args()
    run_dir = run_two_level_vqvae_training(
        config_path=args.config,
        resume_checkpoint=args.resume_checkpoint,
    )
    print('Training completed. Artifacts saved to:', run_dir)


if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    main()
