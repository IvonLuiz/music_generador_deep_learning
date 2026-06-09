import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation import HierarchicalVQVAEReconstructionConfig, HierarchicalVQVAEReconstructionEvaluator
from generation.audio_inversion import AudioInversionConfig
from generation.audio_inversion_cli import add_audio_inversion_args


def _audio_config_from_args(args) -> AudioInversionConfig:
    """!
    @brief Build audio inversion settings for hierarchical VQ-VAE reconstruction.
    @param args Parsed CLI args.
    @return AudioInversionConfig.
    """
    use_fixed = args.use_fixed_db_scale or not args.min_max_values_path
    return AudioInversionConfig(
        method=args.audio_method,
        gradient_steps=args.gradient_inversion_steps,
        gradient_lr=args.gradient_inversion_lr,
        gradient_chunk_frames=args.gradient_inversion_chunk_frames,
        gradient_overlap_frames=args.gradient_inversion_overlap_frames,
        decorsiere_alpha=args.decorsiere_alpha,
        decorsiere_lr=args.decorsiere_lr,
        decorsiere_history_size=args.decorsiere_history_size,
        min_max_values_path=args.min_max_values_path,
        use_fixed_db_scale=use_fixed,
        fixed_min_db=args.fixed_min_db,
        fixed_max_db=args.fixed_max_db,
    )


def main() -> None:
    """!
    @brief CLI entry point for two-level VQ-VAE reconstruction evaluation.
    """
    parser = argparse.ArgumentParser(description="Test hierarchical VQ-VAE reconstruction")
    parser.add_argument("--model_path", type=str, required=True, help="Path to hierarchical VQ-VAE run dir or checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "val", "test", "all"], help="Raw-audio split to sample when the model uses dataset.input_mode=audio")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples to reconstruct")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    parser.add_argument("--save_root", type=str, default="samples/vq_vae_hierarchical_test", help="Root folder for generated artifacts")
    add_audio_inversion_args(parser, include_denormalization=True)
    args = parser.parse_args()

    if args.n_samples <= 0:
        raise ValueError(f"--n_samples must be > 0, got {args.n_samples}")

    config = HierarchicalVQVAEReconstructionConfig(
        model_path=args.model_path,
        n_samples=args.n_samples,
        seed=args.seed,
        min_db=args.fixed_min_db,
        max_db=args.fixed_max_db,
        min_max_values_path=args.min_max_values_path,
        split=args.split,
        save_root=args.save_root,
    )
    result = HierarchicalVQVAEReconstructionEvaluator(config, _audio_config_from_args(args)).run()
    print(f"Saved hierarchical VQ-VAE reconstruction outputs to {result.output_dir}")


if __name__ == "__main__":
    main()
