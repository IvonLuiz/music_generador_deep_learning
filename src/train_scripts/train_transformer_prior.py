import argparse
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.runners import run_jukebox_transformer_prior_training


def main():
    parser = argparse.ArgumentParser(description='Train Jukebox Transformer priors (top/middle/bottom).')
    parser.add_argument('--config', type=str, default='./config/config_transformer_prior.yaml')
    parser.add_argument(
        '--level',
        type=str,
        choices=['top', 'middle', 'bottom'],
        default=None,
        help='Override model.selected_level in config.',
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
    run_dir = run_jukebox_transformer_prior_training(
        config_path=args.config,
        level_override=args.level,
        resume_checkpoint=args.resume_checkpoint,
    )
    print('Transformer prior training complete. Artifacts saved to:', run_dir)


if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    main()
