import os
from typing import Dict, List, Tuple

import torch

from utils import load_config
from modeling.torch.pixel_cnn_hierarchical import HierarchicalCondGatedPixelCNN


def resolve_model_paths(model_dir_or_file: str, weights_file: str = "best_model.pth") -> Tuple[str, str]:
    if os.path.isfile(model_dir_or_file):
        filename = os.path.basename(model_dir_or_file).lower()
        parent_dir = os.path.dirname(model_dir_or_file)

        # Allow passing config file directly
        if filename in ("config.yaml", "config.yml"):
            config_path = model_dir_or_file
            model_path = os.path.join(parent_dir, weights_file)
        else:
            config_path = os.path.join(parent_dir, "config.yaml")
            model_path = model_dir_or_file
    else:
        config_path = os.path.join(model_dir_or_file, "config.yaml")
        model_path = os.path.join(model_dir_or_file, weights_file)

    if not os.path.exists(model_path):
        # Common fallback names used elsewhere in the project
        for candidate in ("best_model.pth", "best_pixelcnn_model.pth", "model.pth"):
            alt = os.path.join(os.path.dirname(config_path), candidate)
            if os.path.exists(alt):
                model_path = alt
                break

    return config_path, model_path


def _get_prior_cfg(config: dict, name: str) -> dict:
    priors = config.get("priors")
    if priors and name in priors:
        return priors[name]
    return config[name]


def get_prior_names(num_prior_levels: int) -> List[str]:
    if num_prior_levels == 2:
        return ["top_prior", "bottom_prior"]
    if num_prior_levels == 3:
        return ["top_prior", "middle_prior", "bottom_prior"]
    raise ValueError(f"Unsupported num_prior_levels={num_prior_levels}. Expected 2 or 3.")


def parse_prior_arrays(config: dict, num_prior_levels: int) -> Tuple[List[int], List[int], List[int], List[float]]:
    prior_names = get_prior_names(num_prior_levels)

    hidden_units: List[int] = []
    num_layers: List[int] = []
    conv_filter_size: List[int] = []
    dropout: List[float] = []

    for name in prior_names:
        prior_cfg = _get_prior_cfg(config, name)
        hidden_units.append(int(prior_cfg["hidden_channels"]))
        num_layers.append(int(prior_cfg["num_layers"]))
        conv_filter_size.append(int(prior_cfg["conv_filter_size"]))
        dropout.append(float(prior_cfg.get("dropout_rate", 0.0)))

    return hidden_units, num_layers, conv_filter_size, dropout


def normalize_state_dict_keys(state_dict: dict) -> dict:
    keys = list(state_dict.keys())
    if not keys:
        return state_dict

    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}

    return state_dict


def infer_num_embeddings_from_state_dict(state_dict: dict, num_prior_levels: int) -> List[int]:
    module_names = ["top_prior", "bottom_level"] if num_prior_levels == 2 else ["top_prior", "middle_level", "bottom_level"]
    result: List[int] = []

    for module_name in module_names:
        candidates = [
            f"{module_name}.embedding.weight",
            f"{module_name}.output_conv.3.weight",
        ]
        num = None
        for key in candidates:
            if key in state_dict:
                num = int(state_dict[key].shape[0])
                break
        result.append(num)

    inferred = [r for r in result if r is not None]
    if not inferred:
        return [512 for _ in module_names]

    fallback = inferred[0]
    return [r if r is not None else fallback for r in result]


def infer_two_level_conditioning_mode(config: dict, state_dict: dict) -> str:
    mode = config.get("model", {}).get("two_level_conditioning_mode", "deconv")
    first_key = "conditioning_stack.0.weight"

    if first_key in state_dict and state_dict[first_key].ndim == 4:
        kernel_h, kernel_w = state_dict[first_key].shape[2], state_dict[first_key].shape[3]
        if (kernel_h, kernel_w) == (4, 4):
            return "deconv"
        if (kernel_h, kernel_w) == (3, 3):
            return "conv"

    return mode


def load_hierarchical_pixelcnn_model(model_dir_or_file: str, device: torch.device, weights_file: str = "best_model.pth"):
    config_path, model_path = resolve_model_paths(model_dir_or_file, weights_file)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    config = load_config(config_path)
    num_prior_levels = int(config.get("model", {}).get("num_prior_levels", 2))

    hidden_units, num_layers, conv_filter_size, dropout = parse_prior_arrays(config, num_prior_levels)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "config" in checkpoint:
        config = checkpoint["config"]
        num_prior_levels = int(config.get("model", {}).get("num_prior_levels", num_prior_levels))
        hidden_units, num_layers, conv_filter_size, dropout = parse_prior_arrays(config, num_prior_levels)

    state_dict = normalize_state_dict_keys(checkpoint["model_state"])
    num_embeddings = infer_num_embeddings_from_state_dict(state_dict, num_prior_levels)

    two_level_conditioning_mode = infer_two_level_conditioning_mode(config, state_dict)

    pixelcnn = HierarchicalCondGatedPixelCNN(
        num_prior_levels=num_prior_levels,
        input_size=[(32, 32)] * num_prior_levels,
        hidden_units=hidden_units,
        num_layers=num_layers,
        conv_filter_size=conv_filter_size,
        dropout=dropout,
        num_embeddings=num_embeddings,
        residual_units=[1024] * num_prior_levels,
        attention_layers=[0] * num_prior_levels,
        attention_heads=[None] * num_prior_levels,
        conditioning_stack_residual_blocks=[None] + [20] * (num_prior_levels - 1),
        two_level_conditioning_mode=two_level_conditioning_mode,
    ).to(device)

    pixelcnn.load_state_dict(state_dict)
    pixelcnn.eval()
    return pixelcnn, config
