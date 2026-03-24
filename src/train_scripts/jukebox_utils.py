
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

from modeling.torch.jukebox_vq_vae import JukeboxVQVAE
from modeling.torch.pixel_cnn_jukebox_levels import JukeboxLevelPixelCNN
from utils import load_config

LEVEL_TO_INT = {'top': 1, 'middle': 2, 'bottom': 3}
LEVEL_TO_PRIOR_CFG = {'top': 'top_prior', 'middle': 'middle_prior', 'bottom': 'bottom_prior'}


def parse_level(level: str) -> str:
    level = str(level).strip().lower()
    if level in ('mid', 'middle'):
        return 'middle'
    if level not in LEVEL_TO_INT:
        raise ValueError("selected_level must be one of: top, middle, bottom")
    return level


def _get_prior_cfg(config: dict, name: str) -> dict:
    priors = config.get('priors')
    if priors and name in priors:
        return priors[name]
    return config[name]


def _normalize_state_dict_keys_for_jukebox(state_dict: dict) -> dict:
    keys = list(state_dict.keys())
    if not keys:
        return state_dict

    candidates = ['module.model.', 'model.module.', 'model.', 'module.']
    for prefix in candidates:
        if all(k.startswith(prefix) for k in keys):
            return {k[len(prefix):]: v for k, v in state_dict.items()}

    model_prefixed = [k for k in keys if k.startswith('model.')]
    if len(model_prefixed) > 0 and len(model_prefixed) >= int(0.8 * len(keys)):
        normalized = {}
        for k, v in state_dict.items():
            normalized[k[len('model.'):]] = v if k.startswith('model.') else v
        return normalized

    return state_dict

def _extract_num_embeddings_from_state_dict(state_dict: dict) -> Tuple[int, Optional[int]]:
    target_num_embeddings = int(state_dict['pixelcnn_prior.embedding.weight'].shape[0])
    cond_key = 'cond_embedding.weight'
    cond_num_embeddings = int(state_dict[cond_key].shape[0]) if cond_key in state_dict else None
    return target_num_embeddings, cond_num_embeddings


def _normalize_state_dict_keys_for_pixelcnn(state_dict: dict) -> dict:
    keys = list(state_dict.keys())
    if not keys:
        return state_dict
    if all(k.startswith('module.') for k in keys):
        return {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict


def _resolve_model_file(model_dir_or_file: str, weights_file: str) -> Tuple[str, str]:
    if os.path.isfile(model_dir_or_file):
        model_file = model_dir_or_file
        config_path = os.path.join(os.path.dirname(model_file), 'config.yaml')
    else:
        model_file = os.path.join(model_dir_or_file, weights_file)
        config_path = os.path.join(model_dir_or_file, 'config.yaml')
    return model_file, config_path


def load_single_level_prior(model_dir_or_file: str, device: torch.device, weights_file: str = 'best_model.pth') -> JukeboxLevelPixelCNN:
    model_file, config_path = _resolve_model_file(model_dir_or_file, weights_file)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found at {config_path}')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f'Weights file not found at {model_file}')

    config = load_config(config_path)
    model_cfg = config.get('model', {})
    selected_level = parse_level(model_cfg.get('selected_level', 'top'))
    prior_cfg = _get_prior_cfg(config, LEVEL_TO_PRIOR_CFG[selected_level])

    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    if 'model_state' not in checkpoint:
        raise KeyError(f"Checkpoint at {model_file} does not contain 'model_state'.")

    state_dict = _normalize_state_dict_keys_for_pixelcnn(checkpoint['model_state'])
    num_embeddings, cond_num_embeddings = _extract_num_embeddings_from_state_dict(state_dict)

    prior = JukeboxLevelPixelCNN(
        level=LEVEL_TO_INT[selected_level],
        hidden_channels=int(prior_cfg['hidden_channels']),
        num_layers=int(prior_cfg['num_layers']),
        conv_filter_size=int(prior_cfg['conv_filter_size']),
        num_embeddings=num_embeddings,
        cond_num_embeddings=cond_num_embeddings,
        two_level_conditioning_mode=model_cfg.get('two_level_conditioning_mode', 'deconv'),
    ).to(device)

    prior.load_state_dict(state_dict)
    prior.eval()
    return prior


def load_jukebox_model(model_dir_or_file: str, level_name: str, device: torch.device, weights_file: str = 'best_model.pth') -> JukeboxVQVAE:
    model_file, config_path = _resolve_model_file(model_dir_or_file, weights_file)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found at {config_path}')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f'Weights file not found at {model_file}')

    config = load_config(config_path)
    model_cfg = config['model']

    level_profiles = model_cfg.get('level_profiles', {})
    if level_name not in level_profiles:
        raise ValueError(f"Level '{level_name}' not found in level_profiles of {config_path}")

    level_profile = level_profiles[level_name]
    levels = level_profile['levels']
    num_residual_layers = level_profile.get('num_residual_layers', 4)

    activation_name = str(model_cfg.get('activation', '')).lower()
    activation_layer = nn.Sigmoid() if activation_name == 'sigmoid' else None

    model = JukeboxVQVAE(
        input_channels=model_cfg['input_channels'],
        hidden_dim=model_cfg['hidden_dim'],
        levels=levels,
        num_residual_layers=num_residual_layers,
        num_embeddings=model_cfg.get('num_embeddings', 2048),
        embedding_dim=model_cfg.get('embedding_dim', 64),
        beta=model_cfg.get('beta', 0.25),
        conv_type=model_cfg.get('conv_type', 2),
        activation_layer=activation_layer,
        dilation_growth_rate=model_cfg.get('dilation_growth_rate', 3),
        channel_growth=model_cfg.get('channel_growth', 1),
    ).to(device)

    print(f'Loading Jukebox {level_name} model from {model_file}')
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    if 'model_state' not in checkpoint:
        raise KeyError(f"Checkpoint at {model_file} does not contain 'model_state'.")

    state_dict = _normalize_state_dict_keys_for_jukebox(checkpoint['model_state'])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        raise RuntimeError(
            f'Error loading Jukebox {level_name} model from {model_file}. '
            f"Missing keys: {missing[:10]}{' ...' if len(missing) > 10 else ''}"
        )
    if unexpected:
        raise RuntimeError(
            f'Error loading Jukebox {level_name} model from {model_file}. '
            f"Unexpected keys: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}"
        )

    model.eval()
    return model