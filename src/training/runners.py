from __future__ import annotations

import copy
from typing import Optional

from utils import load_config

from .adapters import (
    JukeboxTransformerPriorAdapter,
    JukeboxVQVAEAdapter,
    SinglePixelCNNAdapter,
    SingleVQVAEAdapter,
    TwoLevelPixelCNNAdapter,
    TwoLevelVQVAEAdapter,
)
from .data_modules import (
    AudioWindowDataModule,
    JukeboxTransformerPriorDataModule,
    JukeboxVQVAEDataModule,
    QuantizedPriorDataModule,
)
from .engine import TrainingEngine


LEVEL_ALIASES = {'mid': 'middle'}
JUKEBOX_LEVELS = {'top', 'middle', 'bottom'}
VQVAE_LEVEL_TO_TRAINING_KEY = {
    'top': 'top',
    'middle': 'middle',
    'bottom': 'bottom',
}
TRANSFORMER_LEVEL_TO_TRAINING_KEY = {
    'top': 'top_prior',
    'middle': 'middle_prior',
    'bottom': 'bottom_prior',
}


def _load_training_config(config_path: str, family: str, variant: str) -> dict:
    """!
    @brief Load one YAML config and normalize it for TrainingEngine.

    This handles the shared schema only. Level-specific configs are prepared by
    _load_level_training_config so the base loader does not need sentinel values
    or Jukebox-specific branching.
    """
    config = copy.deepcopy(load_config(config_path))
    return _normalize_training_config(config, config_path, family=family, variant=variant)


def _load_level_training_config(
    config_path: str,
    family: str,
    level_override: Optional[str],
    default_level: str,
    training_key_by_level: dict,
    variant_by_level: bool = False,
) -> dict:
    """!
    @brief Load a config that trains one Jukebox level at a time.
    """
    config = copy.deepcopy(load_config(config_path))
    selected = level_override or config.get('model', {}).get('selected_level', default_level)
    level = _parse_jukebox_level(selected)
    variant = level if variant_by_level else 'single_level'
    training_key = training_key_by_level[level]

    config = _normalize_training_config(
        config,
        config_path,
        family=family,
        variant=variant,
        level=level,
        level_training_key=training_key,
    )
    config.setdefault('model', {})['selected_level'] = level
    return config


def _normalize_training_config(
    config: dict,
    config_path: str,
    family: str,
    variant: str,
    level: Optional[str] = None,
    level_training_key: Optional[str] = None,
) -> dict:
    """!
    @brief Merge shared training fields, resume settings, callbacks, and task metadata.
    """
    training_cfg = dict(config.get('training', {}))
    if level_training_key is not None:
        level_training_cfg = dict(training_cfg.get(level_training_key, {}))
        training_cfg.update(level_training_cfg)
    training_cfg['config_path'] = config_path
    config['training'] = training_cfg

    task_cfg = config.setdefault('task', {})
    task_cfg.update({'family': family, 'variant': variant})
    if level is not None:
        task_cfg['level'] = level
    task_cfg.setdefault('name', config.get('model', {}).get('name'))

    config.setdefault('resume', {})
    config.setdefault('callbacks', {})

    callbacks_cfg = config['callbacks']
    callbacks_cfg.setdefault('early_stopping_patience', training_cfg.get('early_stopping_patience'))
    callbacks_cfg.setdefault('save_every', training_cfg.get('save_every', 0))
    callbacks_cfg.setdefault('save_samples', family == 'vqvae')
    callbacks_cfg.setdefault('sample_count', 4)

    resume_cfg = config['resume']
    if 'enabled' not in resume_cfg:
        resume_cfg['enabled'] = bool(training_cfg.get('retrain', False))
    if not resume_cfg.get('checkpoint_path') and training_cfg.get('pretrained_weights_path'):
        resume_cfg['checkpoint_path'] = training_cfg.get('pretrained_weights_path')
    resume_cfg.setdefault('reset_optimizer', bool(training_cfg.get('reset_optimizer', False)))
    resume_cfg.setdefault('reset_scheduler', bool(training_cfg.get('reset_scheduler', resume_cfg['reset_optimizer'])))
    return config


def _parse_jukebox_level(level: str) -> str:
    level = LEVEL_ALIASES.get(str(level).strip().lower(), str(level).strip().lower())
    if level not in JUKEBOX_LEVELS:
        raise ValueError('level must be one of: top, middle, bottom')
    return level


def run_single_vqvae_training(
    config_path: str,
    resume_checkpoint: Optional[str] = None,
) -> str:
    """!
    @brief Train the single-level VQ-VAE using the shared engine.
    """
    config = _load_training_config(config_path, family='vqvae', variant='single')
    if resume_checkpoint:
        config['resume']['enabled'] = True
        config['resume']['checkpoint_path'] = resume_checkpoint
    data_module = AudioWindowDataModule(config)

    return TrainingEngine(
        config=config,
        adapter=SingleVQVAEAdapter(),
        data_module=data_module,
        config_path=config_path,
    ).run()


def run_two_level_vqvae_training(
    config_path: str,
    resume_checkpoint: Optional[str] = None,
) -> str:
    """!
    @brief Train the two-level VQ-VAE using the shared engine.
    """
    config = _load_training_config(config_path, family='vqvae', variant='two_level')
    if resume_checkpoint:
        config['resume']['enabled'] = True
        config['resume']['checkpoint_path'] = resume_checkpoint
    data_module = AudioWindowDataModule(config)

    return TrainingEngine(
        config=config,
        adapter=TwoLevelVQVAEAdapter(),
        data_module=data_module,
        config_path=config_path,
    ).run()


def run_single_pixelcnn_training(
    config_path: str,
    resume_checkpoint: Optional[str] = None,
) -> str:
    """!
    @brief Train a single-level PixelCNN prior on precomputed quantized indices.
    """
    config = _load_training_config(config_path, family='pixelcnn', variant='single')
    if resume_checkpoint:
        config['resume']['enabled'] = True
        config['resume']['checkpoint_path'] = resume_checkpoint
    data_module = QuantizedPriorDataModule(
        config=config,
        variant='single',
    )
    return TrainingEngine(
        config=config,
        adapter=SinglePixelCNNAdapter(),
        data_module=data_module,
        config_path=config_path,
    ).run()


def run_two_level_pixelcnn_training(
    config_path: str,
    resume_checkpoint: Optional[str] = None,
) -> str:
    """!
    @brief Train a two-level hierarchical PixelCNN prior on precomputed indices.
    """
    config = _load_training_config(config_path, family='pixelcnn', variant='two_level')
    if resume_checkpoint:
        config['resume']['enabled'] = True
        config['resume']['checkpoint_path'] = resume_checkpoint
    data_module = QuantizedPriorDataModule(
        config=config,
        variant='two_level',
    )
    return TrainingEngine(
        config=config,
        adapter=TwoLevelPixelCNNAdapter(),
        data_module=data_module,
        config_path=config_path,
    ).run()


def run_jukebox_vqvae_training(
    config_path: str,
    level_override: Optional[str] = None,
    resume_checkpoint: Optional[str] = None,
) -> str:
    """!
    @brief Train one Jukebox VQ-VAE level through the shared engine.
    """
    config = _load_level_training_config(
        config_path,
        family='jukebox_vqvae',
        level_override=level_override,
        default_level='bottom',
        training_key_by_level=VQVAE_LEVEL_TO_TRAINING_KEY,
    )
    level = config['task']['level']
    config['callbacks'].setdefault('save_samples', True)

    if resume_checkpoint:
        config['resume']['enabled'] = True
        config['resume']['checkpoint_path'] = resume_checkpoint

    return TrainingEngine(
        config=config,
        adapter=JukeboxVQVAEAdapter(),
        data_module=JukeboxVQVAEDataModule(config, level=level),
        config_path=config_path,
    ).run()


def run_jukebox_transformer_prior_training(
    config_path: str,
    level_override: Optional[str] = None,
    resume_checkpoint: Optional[str] = None,
) -> str:
    """!
    @brief Train one Jukebox Transformer prior level through the shared engine.
    """
    config = _load_level_training_config(
        config_path,
        family='jukebox_transformer_prior',
        level_override=level_override,
        default_level='top',
        training_key_by_level=TRANSFORMER_LEVEL_TO_TRAINING_KEY,
        variant_by_level=True,
    )
    level = config['task']['level']
    config['callbacks'].setdefault('save_samples', False)

    if resume_checkpoint:
        config['resume']['enabled'] = True
        config['resume']['checkpoint_path'] = resume_checkpoint

    return TrainingEngine(
        config=config,
        adapter=JukeboxTransformerPriorAdapter(),
        data_module=JukeboxTransformerPriorDataModule(config, level=level),
        config_path=config_path,
    ).run()
