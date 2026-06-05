import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation import BottomConditionedPriorConfig, BottomConditionedPriorEvaluator
from generation.audio_inversion import AudioInversionConfig


def build_parser() -> argparse.ArgumentParser:
    """!
    @brief Build bottom-conditioned prior CLI parser.
    @return Configured parser.
    """
    parser = argparse.ArgumentParser(description='Bottom prior test with real conditioning.')
    parser.add_argument('--bottom_prior', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--file', default=None)
    parser.add_argument('--bottom_vqvae', default=None)
    parser.add_argument('--weights_file', default='best_model.pth')
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_root', default='./samples/bottom_prior_conditioned')
    parser.add_argument('--audio_method', default='gradient')
    parser.add_argument('--full_length', action='store_true')
    parser.add_argument('--progress_interval', type=int, default=128)
    return parser


def main(argv=None) -> str:
    """!
    @brief CLI entry point for bottom-conditioned prior evaluation.
    @param argv Optional argv override for tests.
    @return Output directory.
    """
    args = build_parser().parse_args(argv)
    config = BottomConditionedPriorConfig(
        bottom_prior=args.bottom_prior,
        data_root=args.data_root,
        file=args.file,
        bottom_vqvae=args.bottom_vqvae,
        weights_file=args.weights_file,
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        output_root=args.output_root,
        full_length=args.full_length,
        progress_interval=args.progress_interval,
    )
    audio_config = AudioInversionConfig(method=args.audio_method, use_fixed_db_scale=True)
    result = BottomConditionedPriorEvaluator(config, audio_config).run()
    return result.output_dir


if __name__ == '__main__':
    main()
