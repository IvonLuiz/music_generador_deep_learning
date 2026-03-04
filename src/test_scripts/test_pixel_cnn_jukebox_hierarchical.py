import os
import argparse
from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generation.soundgenerator import SoundGenerator
from generation.generate import save_multiple_signals
from modeling.torch.jukebox_vq_vae import JukeboxVQVAE
from processing.preprocess_audio import TARGET_TIME_FRAMES
from test_scripts.hierarchical_pixelcnn_common import (
    load_hierarchical_pixelcnn_model,
    get_prior_names,
    resolve_model_paths,
)
from utils import load_config


def _normalize_state_dict_keys_for_jukebox(state_dict: dict) -> dict:
    keys = list(state_dict.keys())
    if not keys:
        return state_dict

    candidates = ["module.model.", "model.module.", "model.", "module."]
    for prefix in candidates:
        if all(k.startswith(prefix) for k in keys):
            return {k[len(prefix):]: v for k, v in state_dict.items()}

    return state_dict


def _load_jukebox_level_model(model_dir_or_file: str, level_name: str, device: torch.device, weights_file: str = "best_model.pth") -> JukeboxVQVAE:
    config_path, model_path = resolve_model_paths(model_dir_or_file, weights_file)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Weights file not found at {model_path}")

    config = load_config(config_path)
    model_cfg = config["model"]

    level_profiles = model_cfg.get("level_profiles", {})
    if level_name not in level_profiles:
        raise ValueError(f"Level '{level_name}' not found in level_profiles in {config_path}")

    level_profile = level_profiles[level_name]
    levels = level_profile["levels"]
    num_residual_layers = level_profile.get("num_residual_layers", 4)

    activation_name = str(model_cfg.get("activation", "")).lower()
    activation_layer = nn.Sigmoid() if activation_name == "sigmoid" else None

    model = JukeboxVQVAE(
        input_channels=model_cfg["input_channels"],
        hidden_dim=model_cfg["hidden_dim"],
        levels=levels,
        num_residual_layers=num_residual_layers,
        num_embeddings=model_cfg.get("num_embeddings", 2048),
        embedding_dim=model_cfg.get("embedding_dim", 64),
        beta=model_cfg.get("beta", 0.25),
        conv_type=model_cfg.get("conv_type", 2),
        activation_layer=activation_layer,
        dilation_growth_rate=model_cfg.get("dilation_growth_rate", 3),
        channel_growth=model_cfg.get("channel_growth", 1),
    ).to(device)

    print(f"Loading Jukebox {level_name} model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state" not in checkpoint:
        raise KeyError(f"Checkpoint at {model_path} does not contain 'model_state'.")

    state_dict = _normalize_state_dict_keys_for_jukebox(checkpoint["model_state"])
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


def _generate_hierarchical_codes(pixelcnn_model, latent_shapes: List[Tuple[int, int]], num_samples: int, device: torch.device):
    generated_levels = []

    for idx, shape in enumerate(latent_shapes):
        h, w = shape
        if idx == 0:
            print(f"Generating top level ({num_samples}, {h}, {w})...")
            curr = pixelcnn_model.generate(shape=(num_samples, 1, h, w), level="top").squeeze(1)
        elif idx == 1:
            print(f"Generating middle level ({num_samples}, {h}, {w}) conditioned on top...")
            curr = pixelcnn_model.generate(shape=(num_samples, 1, h, w), cond=generated_levels[0], level="mid").squeeze(1)
        else:
            print(f"Generating bottom level ({num_samples}, {h}, {w}) conditioned on middle...")
            curr = pixelcnn_model.generate(shape=(num_samples, 1, h, w), cond=generated_levels[1], level="bottom").squeeze(1)

        generated_levels.append(curr.to(device))

    return generated_levels


def _decode_bottom_from_indices(bottom_indices: torch.Tensor, bottom_model: JukeboxVQVAE):
    bottom_model.eval()
    with torch.no_grad():
        z_q_bottom = F.embedding(bottom_indices, bottom_model.vq.embedding).permute(0, 3, 1, 2).contiguous()
        x_recon = bottom_model.decoder(z_q_bottom)
        if bottom_model.activation_layer is not None:
            x_recon = bottom_model.activation_layer(x_recon)
        spectrograms = x_recon.permute(0, 2, 3, 1).cpu().numpy()
    return spectrograms


def test_jukebox_hierarchical_pixelcnn(pixelcnn_path: str, num_samples: int = 3, min_db: float = -80.0, max_db: float = 0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading hierarchical PixelCNN from {pixelcnn_path}")
    pixelcnn, pixelcnn_config = load_hierarchical_pixelcnn_model(pixelcnn_path, device)

    num_prior_levels = int(pixelcnn_config.get("model", {}).get("num_prior_levels", 3))
    prior_names = get_prior_names(num_prior_levels)
    level_names = [name.replace("_prior", "") for name in prior_names]

    vqvae_cfg = pixelcnn_config.get("vqvae", {})
    level_to_path = {
        "top": vqvae_cfg.get("top_model_dir"),
        "middle": vqvae_cfg.get("middle_model_dir"),
        "bottom": vqvae_cfg.get("bottom_model_dir"),
    }
    weights_file = vqvae_cfg.get("weights_file", "best_model.pth")

    jukebox_models: List[JukeboxVQVAE] = []
    for level_name in level_names:
        model_path = level_to_path.get(level_name)
        if not model_path:
            raise ValueError(f"Missing vqvae path for level '{level_name}' in pixelcnn config")
        jukebox_models.append(_load_jukebox_level_model(model_path, level_name, device, weights_file))

    latent_shapes = _infer_level_shapes(jukebox_models, device)
    print(f"Latent shapes by level ({level_names}): {latent_shapes}")

    generated_codes = _generate_hierarchical_codes(pixelcnn, latent_shapes, num_samples, device)

    print("Decoding bottom-level indices with bottom Jukebox decoder...")
    bottom_specs = _decode_bottom_from_indices(generated_codes[-1], jukebox_models[-1])

    hop_length = int(pixelcnn_config.get("dataset", {}).get("hop_length", 256))
    sound_generator = SoundGenerator(jukebox_models[-1], hop_length=hop_length)
    min_max_values = [{"min": min_db, "max": max_db} for _ in range(num_samples)]

    signals = sound_generator.convert_spectrograms_to_audio(bottom_specs, min_max_values)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"samples/pixelcnn_jukebox_hierarchical_generated/{current_time}/"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving generated audio to {save_dir}")
    save_multiple_signals({"generated": signals}, save_dir)

    spec_dir = os.path.join(save_dir, "spectrograms")
    os.makedirs(spec_dir, exist_ok=True)

    for i, spec in enumerate(bottom_specs):
        plt.figure(figsize=(10, 4))
        plt.imshow(spec[:, :, 0], origin="lower", aspect="auto")
        plt.colorbar()
        plt.title(f"Generated Jukebox Hierarchical {i}")
        plt.savefig(os.path.join(spec_dir, f"sample_{i}.png"))
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pixelcnn", type=str, required=True, help="Path to hierarchical PixelCNN model directory or .pth")
    parser.add_argument("--n_samples", type=int, default=3)
    parser.add_argument("--min_db", type=float, default=-80.0)
    parser.add_argument("--max_db", type=float, default=0.0)
    args = parser.parse_args()

    test_jukebox_hierarchical_pixelcnn(args.pixelcnn, args.n_samples, args.min_db, args.max_db)
