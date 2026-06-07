import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation import SingleVQVAEReconstructionConfig, SingleVQVAEReconstructionEvaluator
from generation.audio_inversion import AudioInversionConfig
from generation.audio_inversion_cli import add_audio_inversion_args
from utils import load_config


def _resolve_default_model_path(config_path: str) -> str:
    """!
    @brief Resolve the configured VQ-VAE run directory used by the legacy test script.
    @param config_path Path to `config_vqvae.yaml`.
    @return VQ-VAE run directory.
    """
    config = load_config(config_path)
    run_dir_name = str(config.get("testing", {}).get("specific_run_dir", "")).strip()
    base_save_dir = str(config.get("training", {}).get("save_dir", "./models/vq_vae"))
    if not run_dir_name:
        raise ValueError("No --model_path was provided and testing.specific_run_dir is missing from the config.")
    return os.path.join(base_save_dir, run_dir_name)


def _audio_config_from_args(args) -> AudioInversionConfig:
    """!
    @brief Build audio inversion settings for VQ-VAE reconstruction.
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
    @brief CLI entry point for single-level VQ-VAE reconstruction evaluation.
    """
    parser = argparse.ArgumentParser(description="Test single-level VQ-VAE reconstruction")
    parser.add_argument("--config", type=str, default="./config/config_vqvae.yaml", help="Training config used to resolve defaults")
    parser.add_argument("--model_path", type=str, default=None, help="Path to VQ-VAE run dir, config.yaml, or checkpoint")
    parser.add_argument("--weights_file", type=str, default=None, help="Checkpoint filename when --model_path is a directory")
    parser.add_argument("--spectrograms_path", type=str, default=None, help="Optional spectrogram dataset override")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples to reconstruct")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    parser.add_argument("--save_root", type=str, default="samples/vqvae_reconstruction", help="Root folder for generated artifacts")
    add_audio_inversion_args(parser, include_denormalization=True)
    args = parser.parse_args()

    if args.n_samples <= 0:
        raise ValueError(f"--n_samples must be > 0, got {args.n_samples}")

    config = load_config(args.config)
    model_path = args.model_path or _resolve_default_model_path(args.config)
    weights_file = args.weights_file or config.get("testing", {}).get("weights_file_choice", "best_model.pth")

    eval_config = SingleVQVAEReconstructionConfig(
        model_path=model_path,
        weights_file=weights_file,
        spectrograms_path=args.spectrograms_path,
        n_samples=args.n_samples,
        seed=args.seed,
        min_db=args.fixed_min_db,
        max_db=args.fixed_max_db,
        min_max_values_path=args.min_max_values_path,
        save_root=args.save_root,
    )
    result = SingleVQVAEReconstructionEvaluator(eval_config, _audio_config_from_args(args)).run()
    print(f"Saved VQ-VAE reconstruction outputs to {result.output_dir}")


if __name__ == "__main__":
    main()
