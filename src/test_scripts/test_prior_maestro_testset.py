import argparse
import os
import sys
from typing import List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation import MaestroPriorTestSetConfig, MaestroPriorTestSetEvaluator, PriorModelTestSpec


def _default_name(kind: str, path: str, level: Optional[str] = None) -> str:
    """!
    @brief Build a stable display name for one prior checkpoint.
    @param kind Prior implementation kind.
    @param path Prior run directory, config, or checkpoint path.
    @param level Optional prior level.
    @return Human-readable model name.
    """
    normalized = os.path.normpath(path)
    base = os.path.basename(normalized)
    if base.lower() in ("config.yaml", "config.yml") or base.endswith(".pth"):
        base = os.path.basename(os.path.dirname(normalized))
    parts = [kind]
    if level:
        parts.append(level)
    parts.append(base or "model")
    return "_".join(parts)


def _append_specs(
    specs: List[PriorModelTestSpec],
    paths: Optional[List[str]],
    kind: str,
    weights_file: str,
    level: Optional[str] = None,
) -> None:
    """!
    @brief Append repeatable CLI model paths to the evaluation spec list.
    @param specs Destination model spec list.
    @param paths CLI paths.
    @param kind Prior implementation kind.
    @param weights_file Checkpoint filename for run-directory inputs.
    @param level Optional prior level.
    """
    for path in paths or []:
        specs.append(
            PriorModelTestSpec(
                name=_default_name(kind, path, level),
                kind=kind,
                model_path=path,
                weights_file=weights_file,
                level=level,
            )
        )


def _build_model_specs(args) -> List[PriorModelTestSpec]:
    """!
    @brief Convert parsed model path flags into prior test specs.
    @param args Parsed CLI arguments.
    @return Prior model specs.
    """
    specs: List[PriorModelTestSpec] = []
    _append_specs(specs, args.single_pixelcnn, "single_pixelcnn", args.weights_file)
    _append_specs(specs, args.hierarchical_pixelcnn, "hierarchical_pixelcnn", args.weights_file)
    _append_specs(specs, args.transformer_top, "transformer", args.weights_file, "top")
    _append_specs(specs, args.transformer_middle, "transformer", args.weights_file, "middle")
    _append_specs(specs, args.transformer_bottom, "transformer", args.weights_file, "bottom")
    return specs


def build_parser() -> argparse.ArgumentParser:
    """!
    @brief Build CLI parser for prior test-set metrics.
    @return Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate PixelCNN/Transformer prior NLL, cross-entropy, bits/token, perplexity, and accuracy."
    )
    parser.add_argument("--single_pixelcnn", action="append", default=[], help="Single-level PixelCNN prior run dir/config/checkpoint. Repeatable.")
    parser.add_argument("--hierarchical_pixelcnn", action="append", default=[], help="Two-level PixelCNN prior run dir/config/checkpoint. Repeatable.")
    parser.add_argument("--transformer_top", action="append", default=[], help="Top Transformer prior run dir/config/checkpoint. Repeatable.")
    parser.add_argument("--transformer_middle", action="append", default=[], help="Middle Transformer prior run dir/config/checkpoint. Repeatable.")
    parser.add_argument("--transformer_bottom", action="append", default=[], help="Bottom Transformer prior run dir/config/checkpoint. Repeatable.")
    parser.add_argument("--weights_file", type=str, default="best_model.pth", help="Checkpoint filename for run-directory inputs.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"], help="Held-out split to score.")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for a quick metric run.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used when shuffling/capping samples.")
    parser.add_argument("--window_parity", type=str, default="all", choices=["all", "even", "odd"], help="Overlap-window subset to score.")
    parser.add_argument("--pixelcnn_quantized_path", type=str, default=None, help="Override single PixelCNN quantized token directory.")
    parser.add_argument("--hierarchical_pixelcnn_quantized_path", type=str, default=None, help="Override hierarchical PixelCNN quantized token directory.")
    parser.add_argument("--transformer_quantized_path", type=str, default=None, help="Override Transformer/Jukebox quantized token directory.")
    parser.add_argument("--processed_path", type=str, default=None, help="Override processed spectrogram path for Transformer split filtering.")
    parser.add_argument("--metadata_path", type=str, default=None, help="Override MAESTRO metadata CSV path for Transformer split filtering.")
    parser.add_argument("--raw_path", type=str, default=None, help="Override MAESTRO raw audio root for Transformer split filtering.")
    parser.add_argument("--save_root", type=str, default="samples/maestro_prior_testset", help="Root folder for prior metric reports.")
    return parser


def main(argv=None) -> str:
    """!
    @brief CLI entry point for prior test-set metrics.
    @param argv Optional argument list override.
    @return Output directory.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError(f"--batch_size must be > 0, got {args.batch_size}.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError(f"--max_samples must be > 0, got {args.max_samples}.")

    specs = _build_model_specs(args)
    if not specs:
        raise ValueError("Provide at least one prior path, for example --transformer_top ./models/transformer_prior/<run>.")

    config = MaestroPriorTestSetConfig(
        models=specs,
        split=args.split,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        seed=args.seed,
        window_parity=args.window_parity,
        pixelcnn_quantized_path=args.pixelcnn_quantized_path,
        hierarchical_pixelcnn_quantized_path=args.hierarchical_pixelcnn_quantized_path,
        transformer_quantized_path=args.transformer_quantized_path,
        processed_path=args.processed_path,
        metadata_path=args.metadata_path,
        raw_path=args.raw_path,
        save_root=args.save_root,
    )
    result = MaestroPriorTestSetEvaluator(config).run()
    print(f"Saved prior test-set metrics to {result.output_dir}")
    return result.output_dir


if __name__ == "__main__":
    main()
