import argparse
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.runners import run_single_vqvae_training


def main():
    parser = argparse.ArgumentParser(description='Train a single-level VQ-VAE.')
    parser.add_argument('--config', type=str, default='./config/config_vqvae.yaml')
    parser.add_argument(
        '--resume-checkpoint',
        type=str,
        default=None,
        help='Resume from a previous model.pth/best_model.pth checkpoint.',
    )
    args = parser.parse_args()
    run_dir = run_single_vqvae_training(
        config_path=args.config,
        resume_checkpoint=args.resume_checkpoint,
    )
    print('Model training complete. Artifacts saved to:', run_dir)


if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    main()
