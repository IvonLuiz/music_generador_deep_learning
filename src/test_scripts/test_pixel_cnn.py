import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation import SinglePixelCNNSamplingConfig, SinglePixelCNNSamplingEvaluator
from generation.audio_inversion import DEFAULT_FIXED_MAX_DB, DEFAULT_FIXED_MIN_DB, AudioInversionConfig
from generation.audio_inversion_cli import add_audio_inversion_args


def _audio_config_from_args(args) -> AudioInversionConfig:
    """!
    @brief Build fixed-dB audio inversion settings for PixelCNN samples.
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
    @brief CLI entry point for single-level PixelCNN sampling evaluation.
    """
    parser = argparse.ArgumentParser(description="Test PixelCNN model")
    parser.add_argument("--vqvae_path", type=str, help="Path to VQ-VAE model")
    parser.add_argument("--pixelcnn_path", type=str, help="Path to PixelCNN model")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--n_samples", type=int, default=None, help="Alias for --num_samples")
    parser.add_argument("--min_db", type=float, default=DEFAULT_FIXED_MIN_DB, help="Minimum dB value")
    parser.add_argument("--max_db", type=float, default=DEFAULT_FIXED_MAX_DB, help="Maximum dB value")
    parser.add_argument("--save_root", type=str, default="samples/pixelcnn_generated", help="Root folder for generated artifacts")
    add_audio_inversion_args(parser, include_denormalization=False)
    args = parser.parse_args()

    n_samples = args.n_samples if args.n_samples is not None else args.num_samples
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    if not os.path.exists(args.vqvae_path):
        raise FileNotFoundError(f"VQ-VAE path does not exist: {args.vqvae_path}")
    if not os.path.exists(args.pixelcnn_path):
        raise FileNotFoundError(f"PixelCNN path does not exist: {args.pixelcnn_path}")

    config = SinglePixelCNNSamplingConfig(
        pixelcnn_path=args.pixelcnn_path,
        vqvae_path=args.vqvae_path,
        n_samples=n_samples,
        min_db=args.min_db,
        max_db=args.max_db,
        save_root=args.save_root,
    )
    result = SinglePixelCNNSamplingEvaluator(config, _audio_config_from_args(args)).run()
    print(f"Saved PixelCNN samples to {result.output_dir}")


if __name__ == "__main__":
    main()
