from __future__ import annotations

import copy
from typing import Optional

from utils import load_config

from .adapters import (
    SinglePixelCNNAdapter,
    SingleVQVAEAdapter,
    TwoLevelPixelCNNAdapter,
    TwoLevelVQVAEAdapter,
)
from .data_modules import QuantizedPriorDataModule, build_vqvae_data_module
from .engine import TrainingEngine


def _prepare_config(config: dict, family: str, variant: str, config_path: str) -> dict:
    config = copy.deepcopy(config)
    task_cfg = config.setdefault('task', {})
    task_cfg.setdefault('family', family)
    task_cfg.setdefault('variant', variant)
    if 'name' not in task_cfg and config.get('model', {}).get('name'):
        task_cfg['name'] = config['model']['name']
    config.setdefault('resume', {})
    config.setdefault('callbacks', {})
    config.setdefault('training', {})['config_path'] = config_path

    training_cfg = config['training']
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


def run_single_vqvae_training(
    config_path: str,
    input_mode_override: Optional[str] = None,
    resume_checkpoint: Optional[str] = None,
) -> str:
    config = _prepare_config(load_config(config_path), 'vqvae', 'single', config_path)
    if input_mode_override:
        config.setdefault('dataset', {})['input_mode'] = input_mode_override
    if resume_checkpoint:
        config['resume']['enabled'] = True
        config['resume']['checkpoint_path'] = resume_checkpoint
    data_module = build_vqvae_data_module(config, input_mode_override=input_mode_override)
    return TrainingEngine(
        config=config,
        adapter=SingleVQVAEAdapter(),
        data_module=data_module,
        config_path=config_path,
    ).run()


def run_two_level_vqvae_training(
    config_path: str,
    input_mode_override: Optional[str] = None,
    resume_checkpoint: Optional[str] = None,
) -> str:
    config = _prepare_config(load_config(config_path), 'vqvae', 'two_level', config_path)
    if input_mode_override:
        config.setdefault('dataset', {})['input_mode'] = input_mode_override
    if resume_checkpoint:
        config['resume']['enabled'] = True
        config['resume']['checkpoint_path'] = resume_checkpoint
    data_module = build_vqvae_data_module(config, input_mode_override=input_mode_override)
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
    config = _prepare_config(load_config(config_path), 'pixelcnn', 'single', config_path)
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
    config = _prepare_config(load_config(config_path), 'pixelcnn', 'two_level', config_path)
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
