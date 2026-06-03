import argparse
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.runners import run_single_pixelcnn_training


def train_pixel_cnn(pixelcnn_config_path: str):
    return run_single_pixelcnn_training(config_path=pixelcnn_config_path)


def main():
    parser = argparse.ArgumentParser(description='Train PixelCNN on single VQ-VAE code indices.')
    parser.add_argument('--config', type=str, default='./config/config_pixelcnn.yaml')
    parser.add_argument(
        '--resume-checkpoint',
        type=str,
        default=None,
        help='Resume from a previous PixelCNN checkpoint.',
    )
    args = parser.parse_args()
    run_dir = run_single_pixelcnn_training(
        config_path=args.config,
        resume_checkpoint=args.resume_checkpoint,
    )
    print('Training complete. Artifacts saved to:', run_dir)


if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    main()
