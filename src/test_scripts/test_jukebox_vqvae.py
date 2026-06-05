import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation import JukeboxVQVAEReconstructionConfig, JukeboxVQVAEReconstructionEvaluator
from generation.audio_inversion import AudioInversionConfig
from generation.audio_inversion_cli import add_audio_inversion_args


def _audio_config_from_args(args) -> AudioInversionConfig:
    """!
    @brief Build audio inversion settings for Jukebox VQ-VAE reconstruction.
    @param args Parsed CLI args.
    @return AudioInversionConfig.
    """
    min_max_path = args.min_max_values_path or args.min_max_values
    use_fixed = args.use_fixed_db_scale or not min_max_path
    return AudioInversionConfig(
        method=args.audio_method,
        gradient_steps=args.gradient_inversion_steps,
        gradient_lr=args.gradient_inversion_lr,
        gradient_chunk_frames=args.gradient_inversion_chunk_frames,
        gradient_overlap_frames=args.gradient_inversion_overlap_frames,
        decorsiere_alpha=args.decorsiere_alpha,
        decorsiere_lr=args.decorsiere_lr,
        decorsiere_history_size=args.decorsiere_history_size,
        min_max_values_path=min_max_path,
        use_fixed_db_scale=use_fixed,
        fixed_min_db=args.fixed_min_db,
        fixed_max_db=args.fixed_max_db,
    )


def main() -> None:
    """!
    @brief CLI entry point for Jukebox VQ-VAE reconstruction evaluation.
    """
    parser = argparse.ArgumentParser(description="Test Jukebox VQ-VAE reconstruction and codebook encoding")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Jukebox VQ-VAE run dir, config.yaml, or .pth")
    parser.add_argument("--level", type=str, default="bottom", help="Jukebox level: top, middle, bottom")
    parser.add_argument("--weights_file", type=str, default="best_model.pth", help="Weights file name if model_path is a dir")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples to test")
    parser.add_argument("--target_time_frames", type=int, default=None, help="Optional time-frame override")
    parser.add_argument("--min_max_values", type=str, default=None, help="Legacy alias for --min_max_values_path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    parser.add_argument("--save_root", type=str, default="samples/jukebox_vqvae_maestro_test", help="Root folder for generated artifacts")
    add_audio_inversion_args(parser, include_denormalization=True)
    args = parser.parse_args()

    if args.n_samples <= 0:
        raise ValueError(f"--n_samples must be > 0, got {args.n_samples}")

    min_max_path = args.min_max_values_path or args.min_max_values
    config = JukeboxVQVAEReconstructionConfig(
        model_path=args.model_path,
        level=args.level,
        weights_file=args.weights_file,
        n_samples=args.n_samples,
        target_time_frames=args.target_time_frames,
        seed=args.seed,
        min_db=args.fixed_min_db,
        max_db=args.fixed_max_db,
        min_max_values_path=min_max_path,
        save_root=args.save_root,
        audio_method=args.audio_method,
    )
    result = JukeboxVQVAEReconstructionEvaluator(config, _audio_config_from_args(args)).run()
    print(f"Saved Jukebox VQ-VAE outputs to {result.output_dir}")


if __name__ == "__main__":
    main()
