from __future__ import annotations

from generation.audio_inversion import (
    DEFAULT_FIXED_MAX_DB,
    DEFAULT_FIXED_MIN_DB,
    SUPPORTED_AUDIO_METHODS,
    AudioInversionConfig,
)


def add_audio_inversion_args(parser, include_denormalization: bool = True) -> None:
    """!
    @brief Add shared spectrogram-to-audio CLI flags.
    @param parser argparse parser or argument group to receive the flags.
    @param include_denormalization Whether to include min/max and fixed-dB flags.
    """
    parser.add_argument(
        '--audio_method',
        type=str,
        default='gradient',
        choices=SUPPORTED_AUDIO_METHODS,
        help='Spectrogram inversion method.',
    )
    parser.add_argument(
        '--gradient_inversion_steps',
        type=int,
        default=1024,
        help='Optimization steps per chunk when --audio_method gradient or decorsiere is used.',
    )
    parser.add_argument(
        '--gradient_inversion_lr',
        type=float,
        default=0.0005,
        help='Adam learning rate when --audio_method gradient is used.',
    )
    parser.add_argument(
        '--gradient_inversion_chunk_frames',
        type=int,
        default=8192,
        help='Spectrogram time frames per gradient/Decorsiere inversion chunk.',
    )
    parser.add_argument(
        '--gradient_inversion_overlap_frames',
        type=int,
        default=2048,
        help='Overlapped spectrogram frames used to crossfade inversion chunks.',
    )
    parser.add_argument(
        '--decorsiere_alpha',
        type=float,
        default=0.3,
        help='Compressed objective exponent for --audio_method decorsiere.',
    )
    parser.add_argument(
        '--decorsiere_lr',
        type=float,
        default=1.0,
        help='L-BFGS learning rate for --audio_method decorsiere.',
    )
    parser.add_argument(
        '--decorsiere_history_size',
        type=int,
        default=10,
        help='L-BFGS history size for --audio_method decorsiere.',
    )
    if include_denormalization:
        parser.add_argument(
            '--min_max_values_path',
            type=str,
            default=None,
            help='Optional explicit path to min_max_values.pkl.',
        )
        parser.add_argument(
            '--use_fixed_db_scale',
            action='store_true',
            help='Force fixed dB denormalization instead of min_max_values.pkl.',
        )
        parser.add_argument(
            '--fixed_min_db',
            type=float,
            default=DEFAULT_FIXED_MIN_DB,
            help='Fixed dB value mapped from normalized 0.0.',
        )
        parser.add_argument(
            '--fixed_max_db',
            type=float,
            default=DEFAULT_FIXED_MAX_DB,
            help='Fixed dB value mapped from normalized 1.0.',
        )


def audio_inversion_config_from_args(args) -> AudioInversionConfig:
    """!
    @brief Convert shared CLI flags into an AudioInversionConfig.
    @param args argparse namespace with audio inversion fields.
    @return Validated AudioInversionConfig.
    """
    return AudioInversionConfig.from_args(args)
