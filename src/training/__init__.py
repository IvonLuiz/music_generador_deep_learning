"""!
@brief Shared training infrastructure for refactored VQ-VAE and PixelCNN scripts.

The package exposes reusable engines, adapters and data modules so individual
train_scripts files can stay small wrappers around model-specific YAML configs.
"""

from .adapters import (
    SinglePixelCNNAdapter,
    SingleVQVAEAdapter,
    TwoLevelPixelCNNAdapter,
    TwoLevelVQVAEAdapter,
)
from .data_modules import (
    AudioWindowDataModule,
    QuantizedPriorDataModule,
    SpectrogramWindowDataModule,
    build_vqvae_data_module,
)
from .engine import StepResult, TrainingAdapter, TrainingEngine

__all__ = [
    'AudioWindowDataModule',
    'QuantizedPriorDataModule',
    'SinglePixelCNNAdapter',
    'SingleVQVAEAdapter',
    'SpectrogramWindowDataModule',
    'StepResult',
    'TrainingAdapter',
    'TrainingEngine',
    'TwoLevelPixelCNNAdapter',
    'TwoLevelVQVAEAdapter',
    'build_vqvae_data_module',
]
