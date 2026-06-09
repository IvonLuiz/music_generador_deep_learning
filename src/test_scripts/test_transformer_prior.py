import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation import TransformerPriorSamplingConfig, TransformerPriorSamplingEvaluator
from generation.audio_inversion import AudioInversionConfig
from generation.audio_inversion_cli import add_audio_inversion_args
from generation.key_conditioning_cli import add_key_conditioning_args, resolve_key_conditioning


def build_parser() -> argparse.ArgumentParser:
    """!
    @brief Build the Transformer prior evaluation CLI parser.
    @return Configured argparse parser.
    """
    parser = argparse.ArgumentParser(description='Sample top/middle/bottom VQ indices from trained Transformer priors')
    for name in ('top_prior', 'middle_prior', 'bottom_prior'):
        parser.add_argument(
            f'--{name}',
            required=True,
            help=f'Path to Transformer {name} prior run directory, config, or .pth',
        )
    parser.add_argument('--bottom_vqvae', type=str, default=None, help='Path to bottom VQ-VAE run directory, config, or .pth')
    add_audio_inversion_args(parser, include_denormalization=True)
    add_key_conditioning_args(parser)
    parser.add_argument('--weights_file', type=str, default='best_model.pth')
    parser.add_argument('--n_samples', type=int, default=6, help='Number of samples to generate (default: 6)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (default: 1.0)')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k filtering for sampling (0 or negative for no filtering)')
    parser.add_argument('--full_length', action='store_true', help='Generate full length audio')
    parser.add_argument('--full_length_overlap_fraction', type=float, default=0.5, help='Overlap between full-length child windows. 0.5 means 50%% overlap (default: 0.5)')
    parser.add_argument('--timing_duration_seconds', type=float, default=240.0, help='Synthetic song duration for timing conditioning when sampling from the top prior (default: 240s)')
    parser.add_argument('--save_root', default='samples/transformer_prior_maestro', help='Root folder for generated artifacts')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser


def _validate_args(args) -> None:
    """!
    @brief Validate common Transformer prior evaluation CLI arguments.
    @param args Parsed CLI args.
    """
    if args.temperature <= 0:
        raise ValueError(f'--temperature must be > 0, got {args.temperature}')
    if args.top_k is not None and args.top_k < 0:
        raise ValueError(f'--top_k must be >= 0, got {args.top_k}')
    if args.full_length_overlap_fraction < 0.0 or args.full_length_overlap_fraction >= 1.0:
        raise ValueError(f'--full_length_overlap_fraction must be in [0, 1), got {args.full_length_overlap_fraction}')
    if args.timing_duration_seconds <= 0:
        raise ValueError(f'--timing_duration_seconds must be > 0, got {args.timing_duration_seconds}')
    resolve_key_conditioning(args.key, args.key_id)


def main(argv=None) -> str:
    """!
    @brief CLI entry point for Transformer prior evaluation.
    @param argv Optional argv override for tests.
    @return Output directory.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)
    audio_config = AudioInversionConfig.from_args(args)
    key_config = resolve_key_conditioning(args.key, args.key_id)
    if key_config is not None:
        print(f'Using key conditioning: {key_config.key_label} (id={key_config.key_id})')
    evaluator_config = TransformerPriorSamplingConfig(
        top_prior_path=args.top_prior,
        middle_prior_path=args.middle_prior,
        bottom_prior_path=args.bottom_prior,
        bottom_vqvae_path=args.bottom_vqvae,
        weights_file=args.weights_file,
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_k=args.top_k,
        full_length=args.full_length,
        full_length_overlap_fraction=args.full_length_overlap_fraction,
        timing_duration_seconds=args.timing_duration_seconds,
        key_id=None if key_config is None else key_config.key_id,
        key_label=None if key_config is None else key_config.key_label,
        seed=args.seed,
        save_root=args.save_root,
    )
    result = TransformerPriorSamplingEvaluator(evaluator_config, audio_config).run()
    print(f'Saved Transformer prior outputs to {result.output_dir}')
    return result.output_dir


if __name__ == '__main__':
    main()
