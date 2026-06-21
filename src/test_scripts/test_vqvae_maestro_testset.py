import argparse
import os
import sys
from typing import List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation import MaestroVQVAETestSetConfig, MaestroVQVAETestSetEvaluator, VQVAEModelTestSpec


def _default_name(variant: str, path: str, level: Optional[str] = None) -> str:
    """!
    @brief Build a stable display name for a model argument.
    @param variant VQ-VAE implementation variant.
    @param path Model run directory, config, or checkpoint path.
    @param level Optional Jukebox level.
    @return Human-readable model name.
    """
    normalized = os.path.normpath(path)
    base = os.path.basename(normalized)
    if base.lower() in ("config.yaml", "config.yml") or base.endswith(".pth"):
        base = os.path.basename(os.path.dirname(normalized))
    parts = [variant]
    if level:
        parts.append(level)
    parts.append(base or "model")
    return "_".join(parts)


def _append_specs(
    specs: List[VQVAEModelTestSpec],
    paths: Optional[List[str]],
    variant: str,
    weights_file: str,
    level: Optional[str] = None,
) -> None:
    """!
    @brief Add CLI model paths to the evaluation spec list.
    @param specs Destination spec list.
    @param paths Model paths parsed from CLI.
    @param variant VQ-VAE implementation variant.
    @param weights_file Checkpoint filename for run-directory inputs.
    @param level Optional Jukebox level.
    """
    for path in paths or []:
        specs.append(
            VQVAEModelTestSpec(
                name=_default_name(variant, path, level),
                variant=variant,
                model_path=path,
                weights_file=weights_file,
                level=level,
            )
        )


def _build_model_specs(args) -> List[VQVAEModelTestSpec]:
    """!
    @brief Convert parsed model arguments into evaluator specs.
    @param args Parsed CLI args.
    @return List of model test specs.
    """
    specs: List[VQVAEModelTestSpec] = []
    _append_specs(specs, args.single_model, "single", args.weights_file)
    _append_specs(specs, args.hierarchical_model, "hierarchical", args.weights_file)
    _append_specs(specs, args.jukebox_top_model, "jukebox", args.weights_file, "top")
    _append_specs(specs, args.jukebox_middle_model, "jukebox", args.weights_file, "middle")
    _append_specs(specs, args.jukebox_bottom_model, "jukebox", args.weights_file, "bottom")
    return specs


def main() -> None:
    """!
    @brief CLI entry point for MAESTRO test-split VQ-VAE MSE comparison.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate VQ-VAE reconstruction MSE on an official MAESTRO metadata split."
    )
    parser.add_argument("--single_model", action="append", default=[], help="Single-level VQ-VAE run dir/config/checkpoint. Repeatable.")
    parser.add_argument("--hierarchical_model", action="append", default=[], help="Two-level VQ-VAE run dir/config/checkpoint. Repeatable.")
    parser.add_argument("--jukebox_top_model", action="append", default=[], help="Jukebox top-level VQ-VAE run dir/config/checkpoint. Repeatable.")
    parser.add_argument("--jukebox_middle_model", action="append", default=[], help="Jukebox middle-level VQ-VAE run dir/config/checkpoint. Repeatable.")
    parser.add_argument("--jukebox_bottom_model", action="append", default=[], help="Jukebox bottom-level VQ-VAE run dir/config/checkpoint. Repeatable.")
    parser.add_argument("--weights_file", type=str, default="best_model.pth", help="Checkpoint filename for run-directory model paths.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"], help="MAESTRO metadata split to evaluate.")
    parser.add_argument("--spectrograms_path", type=str, default=None, help="Override processed spectrogram directory for all models.")
    parser.add_argument("--metadata_path", type=str, default=None, help="Override MAESTRO metadata CSV path.")
    parser.add_argument("--raw_path", type=str, default=None, help="Override MAESTRO raw audio root used for metadata matching.")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for a quick comparison run.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used when shuffling/capping split files.")
    parser.add_argument("--save_root", type=str, default="samples/maestro_vqvae_testset", help="Root folder for MSE reports.")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError(f"--batch_size must be > 0, got {args.batch_size}.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError(f"--max_samples must be > 0, got {args.max_samples}.")

    specs = _build_model_specs(args)
    if not specs:
        raise ValueError("Provide at least one model path, for example --single_model ./models/vq_vae/<run>.")

    config = MaestroVQVAETestSetConfig(
        models=specs,
        split=args.split,
        spectrograms_path=args.spectrograms_path,
        metadata_path=args.metadata_path,
        raw_path=args.raw_path,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        seed=args.seed,
        save_root=args.save_root,
    )
    result = MaestroVQVAETestSetEvaluator(config).run()
    print(f"Saved MAESTRO {args.split} split VQ-VAE MSE reports to {result.output_dir}")


if __name__ == "__main__":
    main()
