import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation import HierarchicalPixelCNNSamplingConfig, HierarchicalPixelCNNSamplingEvaluator
from generation.audio_inversion import DEFAULT_FIXED_MAX_DB, DEFAULT_FIXED_MIN_DB, AudioInversionConfig
from generation.audio_inversion_cli import add_audio_inversion_args


def _audio_config_from_args(args) -> AudioInversionConfig:
    """!
    @brief Build fixed-dB audio inversion settings for hierarchical PixelCNN samples.
    @param args Parsed CLI args.
    @return AudioInversionConfig.
    """
    return AudioInversionConfig(
        method=args.audio_method,
        gradient_steps=args.gradient_inversion_steps,
        gradient_lr=args.gradient_inversion_lr,
        gradient_chunk_frames=args.gradient_inversion_chunk_frames,
        gradient_overlap_frames=args.gradient_inversion_overlap_frames,
        decorsiere_alpha=args.decorsiere_alpha,
        decorsiere_lr=args.decorsiere_lr,
        decorsiere_history_size=args.decorsiere_history_size,
        use_fixed_db_scale=True,
        fixed_min_db=args.min_db,
        fixed_max_db=args.max_db,
    )


def main() -> None:
    """!
    @brief CLI entry point for hierarchical PixelCNN sampling evaluation.
    """
    parser = argparse.ArgumentParser(description="Test hierarchical PixelCNN model")
    parser.add_argument("--pixelcnn", type=str, required=True, help="Path to hierarchical PixelCNN model directory or .pth")
    parser.add_argument("--vqvae", type=str, required=True, help="Path to hierarchical VQ-VAE model directory or .pth")
    parser.add_argument("--n_samples", type=int, default=3, help="Number of samples to generate")
    parser.add_argument("--min_db", type=float, default=DEFAULT_FIXED_MIN_DB)
    parser.add_argument("--max_db", type=float, default=DEFAULT_FIXED_MAX_DB)
    parser.add_argument("--save_root", type=str, default="samples/pixelcnn_hierarchical_generated", help="Root folder for generated artifacts")
    add_audio_inversion_args(parser, include_denormalization=False)
    args = parser.parse_args()

    if args.n_samples <= 0:
        raise ValueError(f"--n_samples must be > 0, got {args.n_samples}")
    if not os.path.exists(args.vqvae):
        raise FileNotFoundError(f"Hierarchical VQ-VAE path does not exist: {args.vqvae}")
    if not os.path.exists(args.pixelcnn):
        raise FileNotFoundError(f"Hierarchical PixelCNN path does not exist: {args.pixelcnn}")

    config = HierarchicalPixelCNNSamplingConfig(
        pixelcnn_path=args.pixelcnn,
        vqvae_path=args.vqvae,
        n_samples=args.n_samples,
        min_db=args.min_db,
        max_db=args.max_db,
        save_root=args.save_root,
    )
    result = HierarchicalPixelCNNSamplingEvaluator(config, _audio_config_from_args(args)).run()
    print(f"Saved hierarchical PixelCNN samples to {result.output_dir}")


if __name__ == "__main__":
    main()
