import argparse
import os
import sys
from datetime import datetime
from typing import List, Optional, Tuple
import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generation.generate import save_multiple_signals
from generation.soundgenerator import SoundGenerator
from modeling.torch.jukebox_vq_vae import JukeboxVQVAE
from modeling.torch.pixel_cnn_jukebox_levels import JukeboxLevelPixelCNN
from processing.preprocess_audio import TARGET_TIME_FRAMES
from test_scripts.hierarchical_pixelcnn_common import resolve_model_paths
from utils import load_config, load_maestro


LEVEL_TO_INT = {'top': 1, 'middle': 2, 'bottom': 3}
LEVEL_TO_PRIOR_CFG = {'top': 'top_prior', 'middle': 'middle_prior', 'bottom': 'bottom_prior'}
COND_LEVEL = {'top': None, 'middle': 'top', 'bottom': 'middle'}
LATEST_ALIASES = {'latest', 'newest', 'auto', 'most_recent'}


def _normalize_state_dict_keys_for_jukebox(state_dict: dict) -> dict:
    keys = list(state_dict.keys())
    if not keys:
        return state_dict

    candidates = ['module.model.', 'model.module.', 'model.', 'module.']
    for prefix in candidates:
        if all(k.startswith(prefix) for k in keys):
            return {k[len(prefix):]: v for k, v in state_dict.items()}

    return state_dict


def _normalize_state_dict_keys_for_pixelcnn(state_dict: dict) -> dict:
    keys = list(state_dict.keys())
    if not keys:
        return state_dict
    if all(k.startswith('module.') for k in keys):
        return {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict


def _load_jukebox_level_model(model_dir_or_file: str, level_name: str, device: torch.device, weights_file: str = 'best_model.pth') -> JukeboxVQVAE:
    config_path, model_path = resolve_model_paths(model_dir_or_file, weights_file)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found at {config_path}')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Weights file not found at {model_path}')

    config = load_config(config_path)
    model_cfg = config['model']

    level_profiles = model_cfg.get('level_profiles', {})
    if level_name not in level_profiles:
        raise ValueError(f"Level '{level_name}' not found in level_profiles in {config_path}")

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

    print(f'Loading Jukebox {level_name} model from {model_path}')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state' not in checkpoint:
        raise KeyError(f"Checkpoint at {model_path} does not contain 'model_state'.")

    state_dict = _normalize_state_dict_keys_for_jukebox(checkpoint['model_state'])
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _extract_num_embeddings_from_state_dict(state_dict: dict) -> Tuple[int, Optional[int]]:
    target_num_embeddings = int(state_dict['pixelcnn_prior.embedding.weight'].shape[0])
    cond_key = 'cond_embedding.weight'
    cond_num_embeddings = int(state_dict[cond_key].shape[0]) if cond_key in state_dict else None
    return target_num_embeddings, cond_num_embeddings


def _load_single_level_prior(model_dir_or_file: str, level_name: str, device: torch.device, weights_file: str = 'best_model.pth') -> JukeboxLevelPixelCNN:
    config_path, model_path = resolve_model_paths(model_dir_or_file, weights_file)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found at {config_path}')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Weights file not found at {model_path}')

    config = load_config(config_path)
    prior_cfg_name = LEVEL_TO_PRIOR_CFG[level_name]
    prior_cfg = config.get('priors', {}).get(prior_cfg_name)
    if prior_cfg is None:
        raise ValueError(f'Missing priors.{prior_cfg_name} in {config_path}')

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state' not in checkpoint:
        raise KeyError(f"Checkpoint at {model_path} does not contain 'model_state'.")

    state_dict = _normalize_state_dict_keys_for_pixelcnn(checkpoint['model_state'])
    num_embeddings, cond_num_embeddings = _extract_num_embeddings_from_state_dict(state_dict)

    model = JukeboxLevelPixelCNN(
        level=LEVEL_TO_INT[level_name],
        hidden_channels=int(prior_cfg['hidden_channels']),
        num_layers=int(prior_cfg['num_layers']),
        conv_filter_size=int(prior_cfg['conv_filter_size']),
        num_embeddings=num_embeddings,
        cond_num_embeddings=cond_num_embeddings,
        two_level_conditioning_mode=config.get('model', {}).get('two_level_conditioning_mode', 'deconv'),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _infer_level_shapes(jukebox_models: List[JukeboxVQVAE], device: torch.device) -> List[Tuple[int, int]]:
    dummy = torch.zeros((1, 1, TARGET_TIME_FRAMES, TARGET_TIME_FRAMES), device=device)
    shapes: List[Tuple[int, int]] = []

    with torch.no_grad():
        for model in jukebox_models:
            z = model.encoder(dummy)
            z = model.pre_vq_conv(z)
            shapes.append((int(z.shape[2]), int(z.shape[3])))

    return shapes


def _encode_indices_for_level(level_model: JukeboxVQVAE, x_batch: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        z = level_model.encoder(x_batch)
        z = level_model.pre_vq_conv(z)
        _, indices, _, _, _ = level_model.vq(z)
    return indices


def _sample_real_batch(pixelcnn_config: dict, num_samples: int, device: torch.device) -> torch.Tensor:
    dataset_cfg = pixelcnn_config.get('dataset', {})
    processed_path = dataset_cfg.get('processed_path')
    target_time_frames = int(dataset_cfg.get('target_time_frames', TARGET_TIME_FRAMES))
    if processed_path is None:
        raise ValueError('Missing dataset.processed_path in PixelCNN config; required for staged real-code evaluation.')

    x_all, _ = load_maestro(processed_path, target_time_frames)
    if len(x_all) == 0:
        raise ValueError('Dataset is empty; cannot run staged real-code evaluation.')

    actual_n = min(num_samples, len(x_all))
    idx = np.random.choice(len(x_all), actual_n, replace=False)
    x_np = x_all[idx]
    x_t = torch.from_numpy(x_np).permute(0, 3, 1, 2).float().to(device)
    return x_t


def _decode_bottom_from_indices(bottom_indices: torch.Tensor, bottom_model: JukeboxVQVAE):
    bottom_model.eval()
    with torch.no_grad():
        z_q_bottom = F.embedding(bottom_indices, bottom_model.vq.embedding).permute(0, 3, 1, 2).contiguous()
        x_recon = bottom_model.decoder(z_q_bottom)
        if bottom_model.activation_layer is not None:
            x_recon = bottom_model.activation_layer(x_recon)
        spectrograms = x_recon.permute(0, 2, 3, 1).cpu().numpy()
    return spectrograms


def _load_or_generate_top_codes(top_codes_file: Optional[str],
                                top_prior_model: Optional[JukeboxLevelPixelCNN],
                                num_samples: int,
                                top_shape: Tuple[int, int],
                                temperature: float,
                                top_k: int,
                                device: torch.device) -> torch.Tensor:
    top_h, top_w = top_shape
    top_seq_len = int(top_h * top_w)

    if top_codes_file:
        top_np = np.load(top_codes_file)
        if top_np.ndim == 2:
            if top_np.shape[1] != top_seq_len:
                raise ValueError(f'Top code file sequence length {top_np.shape[1]} does not match expected top grid {top_h}x{top_w} ({top_seq_len})')
            top_np = top_np.reshape(top_np.shape[0], top_h, top_w)
        elif top_np.ndim == 3:
            if tuple(top_np.shape[1:]) != (top_h, top_w):
                raise ValueError(f'Top code file grid {tuple(top_np.shape[1:])} does not match expected {(top_h, top_w)}')
        else:
            raise ValueError('Top code file must have shape (B, T) or (B, H, W)')

        actual_n = min(num_samples, top_np.shape[0])
        top_codes = torch.from_numpy(top_np[:actual_n]).long().to(device)
        print(f'Loaded top codes from {top_codes_file} with shape {tuple(top_codes.shape)}')
        return top_codes

    if top_prior_model is None:
        raise ValueError('No top prior available. Provide --top_pixelcnn or use --top_codes_file/--transformer_prior')

    top_k_value = top_k if top_k > 0 else None
    top_codes = top_prior_model.generate(
        shape=(num_samples, 1, top_h, top_w),
        cond=None,
        temperature=temperature,
        top_k=top_k_value,
    ).squeeze(1)
    print(f'Generated top codes from PixelCNN with shape {tuple(top_codes.shape)}')
    return top_codes


def _resolve_prior_model_paths(pixelcnn_cfg: dict,
                               top_pixelcnn: Optional[str],
                               middle_pixelcnn: Optional[str],
                               bottom_pixelcnn: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    models_cfg = pixelcnn_cfg.get('pixelcnn_models', {})
    weights_file = models_cfg.get('weights_file', 'best_model.pth')

    top_path = top_pixelcnn or models_cfg.get('top_model_dir')
    mid_path = middle_pixelcnn or models_cfg.get('middle_model_dir')
    bot_path = bottom_pixelcnn or models_cfg.get('bottom_model_dir')

    if _is_latest_alias(top_path):
        top_path = _resolve_latest_prior_dir(pixelcnn_cfg, 'top')
    if _is_latest_alias(mid_path):
        mid_path = _resolve_latest_prior_dir(pixelcnn_cfg, 'middle')
    if _is_latest_alias(bot_path):
        bot_path = _resolve_latest_prior_dir(pixelcnn_cfg, 'bottom')

    return top_path, mid_path, bot_path, weights_file


def _is_latest_alias(value: Optional[str]) -> bool:
    return isinstance(value, str) and value.strip().lower() in LATEST_ALIASES


def _find_latest_run_dir(parent_dir: str) -> Optional[str]:
    if not os.path.isdir(parent_dir):
        return None

    candidates = []
    for name in os.listdir(parent_dir):
        run_dir = os.path.join(parent_dir, name)
        if not os.path.isdir(run_dir):
            continue
        if not os.path.isfile(os.path.join(run_dir, 'config.yaml')):
            continue
        candidates.append(run_dir)

    if not candidates:
        return None

    candidates.sort()
    latest = candidates[-1]
    candidates_by_mtime = sorted(candidates, key=lambda p: os.path.getmtime(p))
    if os.path.getmtime(candidates_by_mtime[-1]) > os.path.getmtime(latest):
        latest = candidates_by_mtime[-1]
    return latest


def _resolve_latest_prior_dir(config: dict, level_name: str) -> str:
    model_name = str(config.get('model', {}).get('name', '')).strip()
    save_root = str(config.get('training', {}).get('save_dir', './models/')).strip()

    primary_parent = os.path.join(save_root, f'{model_name}_{level_name}_prior') if model_name else None
    if primary_parent:
        latest = _find_latest_run_dir(primary_parent)
        if latest:
            return latest

    pattern = os.path.join(save_root, f'*_{level_name}_prior')
    parent_dirs = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    latest_candidates = []
    for parent in parent_dirs:
        run_dir = _find_latest_run_dir(parent)
        if run_dir:
            latest_candidates.append(run_dir)

    if latest_candidates:
        latest_candidates.sort(key=lambda p: os.path.getmtime(p))
        return latest_candidates[-1]

    raise FileNotFoundError(
        f"Could not resolve latest prior for level={level_name}. "
        f"Expected runs under '{save_root}' matching '*_{level_name}_prior/<timestamp>/'"
    )


def test_jukebox_hierarchical_pixelcnn(
    pixelcnn_config_path: str,
    num_samples: int = 3,
    min_db: float = -80.0,
    max_db: float = 0.0,
    stage_mode: str = 'fully_generated',
    temperature: float = 1.0,
    top_k: int = 0,
    top_codes_file: str = None,
    top_pixelcnn: str = None,
    middle_pixelcnn: str = None,
    bottom_pixelcnn: str = None,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pixelcnn_config = load_config(pixelcnn_config_path)
    if stage_mode not in ('fully_generated', 'real_top', 'real_top_middle'):
        raise ValueError('stage_mode must be one of: fully_generated, real_top, real_top_middle')
    if temperature <= 0:
        raise ValueError(f'temperature must be > 0, got {temperature}')
    if top_k < 0:
        raise ValueError(f'top_k must be >= 0, got {top_k}')

    vqvae_cfg = pixelcnn_config.get('vqvae', {})
    weights_file = vqvae_cfg.get('weights_file', 'best_model.pth')
    top_jbx = _load_jukebox_level_model(vqvae_cfg['top_model_dir'], 'top', device, weights_file)
    mid_jbx = _load_jukebox_level_model(vqvae_cfg['middle_model_dir'], 'middle', device, weights_file)
    bot_jbx = _load_jukebox_level_model(vqvae_cfg['bottom_model_dir'], 'bottom', device, weights_file)
    jukebox_models = [top_jbx, mid_jbx, bot_jbx]

    latent_shapes = _infer_level_shapes(jukebox_models, device)
    print(f'Latent shapes by level ([top, middle, bottom]): {latent_shapes}')

    top_path, mid_path, bot_path, pixelcnn_weights_file = _resolve_prior_model_paths(
        pixelcnn_config, top_pixelcnn, middle_pixelcnn, bottom_pixelcnn
    )

    need_top_prior = stage_mode == 'fully_generated' and not top_codes_file
    need_mid_prior = stage_mode in ('fully_generated', 'real_top') or bool(top_codes_file)
    need_bot_prior = True

    top_prior = _load_single_level_prior(top_path, 'top', device, pixelcnn_weights_file) if need_top_prior else None
    if need_top_prior and top_prior is None:
        raise ValueError('Top prior is required but no top prior path was provided')

    if need_mid_prior and not mid_path:
        raise ValueError('Middle prior is required; provide --middle_pixelcnn or pixelcnn_models.middle_model_dir')
    mid_prior = _load_single_level_prior(mid_path, 'middle', device, pixelcnn_weights_file) if need_mid_prior else None

    if need_bot_prior and not bot_path:
        raise ValueError('Bottom prior is required; provide --bottom_pixelcnn or pixelcnn_models.bottom_model_dir')
    bot_prior = _load_single_level_prior(bot_path, 'bottom', device, pixelcnn_weights_file) if need_bot_prior else None

    generated_codes: List[Optional[torch.Tensor]] = [None, None, None]

    if stage_mode == 'fully_generated':
        generated_codes[0] = _load_or_generate_top_codes(
            top_codes_file=top_codes_file,
            top_prior_model=top_prior,
            num_samples=num_samples,
            top_shape=latent_shapes[0],
            temperature=temperature,
            top_k=top_k,
            device=device,
        )

        top_k_value = top_k if top_k > 0 else None
        h_mid, w_mid = latent_shapes[1]
        generated_codes[1] = mid_prior.generate(
            shape=(generated_codes[0].shape[0], 1, h_mid, w_mid),
            cond=generated_codes[0],
            temperature=temperature,
            top_k=top_k_value,
        ).squeeze(1)

        h_bot, w_bot = latent_shapes[2]
        generated_codes[2] = bot_prior.generate(
            shape=(generated_codes[1].shape[0], 1, h_bot, w_bot),
            cond=generated_codes[1],
            temperature=temperature,
            top_k=top_k_value,
        ).squeeze(1)
    else:
        real_batch = _sample_real_batch(pixelcnn_config, num_samples, device)
        generated_codes[0] = _encode_indices_for_level(top_jbx, real_batch)

        if stage_mode == 'real_top':
            top_k_value = top_k if top_k > 0 else None
            h_mid, w_mid = latent_shapes[1]
            generated_codes[1] = mid_prior.generate(
                shape=(generated_codes[0].shape[0], 1, h_mid, w_mid),
                cond=generated_codes[0],
                temperature=temperature,
                top_k=top_k_value,
            ).squeeze(1)
        else:
            generated_codes[1] = _encode_indices_for_level(mid_jbx, real_batch)

        top_k_value = top_k if top_k > 0 else None
        h_bot, w_bot = latent_shapes[2]
        generated_codes[2] = bot_prior.generate(
            shape=(generated_codes[1].shape[0], 1, h_bot, w_bot),
            cond=generated_codes[1],
            temperature=temperature,
            top_k=top_k_value,
        ).squeeze(1)

    print('Decoding bottom-level indices with bottom Jukebox decoder...')
    bottom_specs = _decode_bottom_from_indices(generated_codes[2], bot_jbx)

    hop_length = int(pixelcnn_config.get('dataset', {}).get('hop_length', 256))
    sound_generator = SoundGenerator(bot_jbx, hop_length=hop_length)
    actual_samples = int(generated_codes[2].shape[0])
    min_max_values = [{'min': min_db, 'max': max_db} for _ in range(actual_samples)]

    signals = sound_generator.convert_spectrograms_to_audio(bottom_specs, min_max_values)

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = f'samples/pixelcnn_jukebox_hierarchical_generated/{current_time}/'
    os.makedirs(save_dir, exist_ok=True)

    print(f'Saving generated audio to {save_dir}')
    save_multiple_signals({'generated': signals}, save_dir)

    spec_dir = os.path.join(save_dir, 'spectrograms')
    os.makedirs(spec_dir, exist_ok=True)

    for i, spec in enumerate(bottom_specs):
        plt.figure(figsize=(10, 4))
        plt.imshow(spec[:, :, 0], origin='lower', aspect='auto')
        plt.colorbar()
        plt.title(f'Generated Jukebox Hierarchical {i}')
        plt.savefig(os.path.join(spec_dir, f'sample_{i}.png'))
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pixelcnn_config', type=str, default='config/config_pixelcnn_jukebox_hierarchical.yaml', help='Config containing dataset/vqvae info and optional pixelcnn_models paths')
    parser.add_argument('--top_pixelcnn', type=str, default=None, help='Top prior run dir or .pth path')
    parser.add_argument('--middle_pixelcnn', type=str, default=None, help='Middle prior run dir or .pth path')
    parser.add_argument('--bottom_pixelcnn', type=str, default=None, help='Bottom prior run dir or .pth path')
    parser.add_argument('--n_samples', type=int, default=3)
    parser.add_argument('--min_db', type=float, default=-40.0)
    parser.add_argument('--max_db', type=float, default=2.0)
    parser.add_argument(
        '--stage_mode',
        type=str,
        default='fully_generated',
        choices=['fully_generated', 'real_top', 'real_top_middle'],
        help='Use staged generation to isolate noise sources across levels.',
    )
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (>0).')
    parser.add_argument('--top_k', type=int, default=0, help='Top-k filtering; 0 disables it.')
    parser.add_argument('--top_codes_file', type=str, default=None, help='Optional .npy file with top codes shaped (B,T) or (B,H,W). If set, top generation uses these codes.')
    args = parser.parse_args()

    test_jukebox_hierarchical_pixelcnn(
        pixelcnn_config_path=args.pixelcnn_config,
        num_samples=args.n_samples,
        min_db=args.min_db,
        max_db=args.max_db,
        stage_mode=args.stage_mode,
        temperature=args.temperature,
        top_k=args.top_k,
        top_codes_file=args.top_codes_file,
        top_pixelcnn=args.top_pixelcnn,
        middle_pixelcnn=args.middle_pixelcnn,
        bottom_pixelcnn=args.bottom_pixelcnn,
    )
