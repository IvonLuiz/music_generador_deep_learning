from __future__ import annotations

import os
from typing import List, Tuple

import torch

from modeling.torch.pixel_cnn_hierarchical import HierarchicalCondGatedPixelCNN
from modeling.torch.pixel_cnn import ConditionalGatedPixelCNN
from train_scripts.jukebox_utils import load_jukebox_model
from utils import (
    load_config,
    load_vqvae_hierarchical_model_wrapper,
    load_vqvae_model,
)


class ModelLoader:
    """!
    @brief Shared model/checkpoint resolution helpers for evaluation scripts.
    """

    @staticmethod
    def resolve_model_paths(model_dir_or_file: str, weights_file: str = "best_model.pth") -> Tuple[str, str]:
        """!
        @brief Resolve config and checkpoint paths from a run directory or file.
        @param model_dir_or_file Run directory, config file, or checkpoint file.
        @param weights_file Preferred checkpoint filename when a directory/config is passed.
        @return Tuple `(config_path, model_path)`.
        """
        requested_weights_file = weights_file
        if os.path.isfile(model_dir_or_file):
            filename = os.path.basename(model_dir_or_file).lower()
            parent_dir = os.path.dirname(model_dir_or_file)
            if filename in ("config.yaml", "config.yml"):
                config_path = model_dir_or_file
                model_path = os.path.join(parent_dir, weights_file)
            else:
                config_path = os.path.join(parent_dir, "config.yaml")
                model_path = model_dir_or_file
        else:
            config_path = os.path.join(model_dir_or_file, "config.yaml")
            model_path = os.path.join(model_dir_or_file, weights_file)

        if not os.path.exists(model_path) and requested_weights_file == "best_model.pth":
            for candidate in ("best_model.pth", "best_pixelcnn_model.pth", "model.pth"):
                alt = os.path.join(os.path.dirname(config_path), candidate)
                if os.path.exists(alt):
                    model_path = alt
                    break
        return config_path, model_path

    @staticmethod
    def load_single_vqvae(model_dir_or_file: str, device: torch.device, weights_file: str = "best_model.pth"):
        """!
        @brief Load a single-level VQ-VAE.
        @param model_dir_or_file Run directory or checkpoint file.
        @param device Target torch device.
        @param weights_file Checkpoint filename.
        @return Loaded model.
        """
        return load_vqvae_model(model_dir_or_file, device, weights_file=weights_file)

    @staticmethod
    def load_hierarchical_vqvae(model_dir_or_file: str, device: torch.device):
        """!
        @brief Load a two-level hierarchical VQ-VAE.
        @param model_dir_or_file Run directory or checkpoint file.
        @param device Target torch device.
        @return Loaded model.
        """
        return load_vqvae_hierarchical_model_wrapper(model_dir_or_file, device)

    @staticmethod
    def load_jukebox_vqvae(model_dir_or_file: str, level: str, device: torch.device, weights_file: str = "best_model.pth"):
        """!
        @brief Load one Jukebox VQ-VAE level.
        @param model_dir_or_file Run directory, config file, or checkpoint file.
        @param level Jukebox level.
        @param device Target torch device.
        @param weights_file Checkpoint filename.
        @return Loaded Jukebox VQ-VAE.
        """
        model_ref = ModelLoader.model_reference(model_dir_or_file)
        return load_jukebox_model(model_ref, level, device, weights_file)

    @staticmethod
    def model_reference(model_dir_or_file: str) -> str:
        """!
        @brief Normalize a model reference so config files resolve to their parent run dir.
        @param model_dir_or_file Run directory, config file, or checkpoint file.
        @return Model reference path.
        """
        if os.path.isfile(model_dir_or_file):
            name = os.path.basename(model_dir_or_file).lower()
            if name in ("config.yaml", "config.yml"):
                return os.path.dirname(model_dir_or_file)
        return model_dir_or_file

    @staticmethod
    def load_single_pixelcnn(
        model_dir_or_file: str,
        device: torch.device,
        num_embeddings: int,
        weights_file: str = None,
    ):
        """!
        @brief Load a single-level PixelCNN.
        @param model_dir_or_file Run directory or checkpoint file.
        @param device Target torch device.
        @param num_embeddings Codebook size.
        @param weights_file Optional checkpoint filename.
        @return Loaded PixelCNN model.
        """
        if os.path.isdir(model_dir_or_file):
            config_path = os.path.join(model_dir_or_file, "config.yaml")
            if not weights_file:
                weights_file = "best_model.pth"
            model_file = os.path.join(model_dir_or_file, weights_file)
        else:
            config_path = os.path.join(os.path.dirname(model_dir_or_file), "config.yaml")
            model_file = model_dir_or_file

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        config = load_config(config_path)
        model_config = config['model']
        
        hidden_channels = model_config['hidden_channels']
        num_layers = model_config['num_layers']
        kernel_size = model_config['kernel_size']
        
        # K (num_embeddings) must be in the config or provided. 
        if num_embeddings is not None:
            K = num_embeddings
        elif 'K' in model_config:
            K = model_config['K']
        elif 'num_embeddings' in model_config:
            K = model_config['num_embeddings']
        else:
            raise ValueError("Model config must contain 'K' or 'num_embeddings' to initialize PixelCNN, or it must be passed as an argument.")

        pixel_cnn = ConditionalGatedPixelCNN(
            in_channels=1,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            num_classes=K,
            num_embeddings=K,
        ).to(device)
        
        print(f"Loading PixelCNN weights from {model_file}")
        checkpoint = torch.load(model_file, map_location=device, weights_only=False)
        pixel_cnn.load_state_dict(checkpoint['model_state'])
        pixel_cnn.eval()
        
        return pixel_cnn

    @staticmethod
    def load_hierarchical_pixelcnn_model(model_dir_or_file: str, device: torch.device, weights_file: str = "best_model.pth"):
        """!
        @brief Load a hierarchical PixelCNN using saved config/checkpoint metadata.
        @param model_dir_or_file Run directory, config file, or checkpoint file.
        @param device Target torch device.
        @param weights_file Checkpoint filename.
        @return Tuple `(model, config)`.
        """
        config_path, model_path = ModelLoader.resolve_model_paths(model_dir_or_file, weights_file)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        config = load_config(config_path)
        num_prior_levels = int(config.get("model", {}).get("num_prior_levels", 2))
        hidden_units, num_layers, conv_filter_size, dropout = ModelLoader.parse_prior_arrays(config, num_prior_levels)

        print(f"Loading Hierarchical PixelCNN from {model_path} with config {config_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if "config" in checkpoint:
            config = checkpoint["config"]
            num_prior_levels = int(config.get("model", {}).get("num_prior_levels", num_prior_levels))
            hidden_units, num_layers, conv_filter_size, dropout = ModelLoader.parse_prior_arrays(config, num_prior_levels)

        state_dict = ModelLoader.normalize_state_dict_keys(checkpoint["model_state"])
        num_embeddings = ModelLoader.infer_num_embeddings_from_state_dict(state_dict, num_prior_levels)
        two_level_conditioning_mode = ModelLoader.infer_two_level_conditioning_mode(config, state_dict)

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

    @staticmethod
    def _get_prior_cfg(config: dict, name: str) -> dict:
        priors = config.get("priors")
        if priors and name in priors:
            return priors[name]
        return config[name]

    @staticmethod
    def get_prior_names(num_prior_levels: int) -> List[str]:
        """!
        @brief Return config section names for a hierarchical PixelCNN.
        @param num_prior_levels Number of prior levels.
        @return Prior section names.
        """
        if num_prior_levels == 2:
            return ["top_prior", "bottom_prior"]
        if num_prior_levels == 3:
            return ["top_prior", "middle_prior", "bottom_prior"]
        raise ValueError(f"Unsupported num_prior_levels={num_prior_levels}. Expected 2 or 3.")

    @staticmethod
    def parse_prior_arrays(config: dict, num_prior_levels: int):
        """!
        @brief Parse per-level PixelCNN hyperparameter arrays.
        @param config PixelCNN config.
        @param num_prior_levels Number of prior levels.
        @return Tuple `(hidden_units, num_layers, conv_filter_size, dropout)`.
        """
        hidden_units: List[int] = []
        num_layers: List[int] = []
        conv_filter_size: List[int] = []
        dropout: List[float] = []
        for name in ModelLoader.get_prior_names(num_prior_levels):
            prior_cfg = ModelLoader._get_prior_cfg(config, name)
            hidden_units.append(int(prior_cfg["hidden_channels"]))
            num_layers.append(int(prior_cfg["num_layers"]))
            conv_filter_size.append(int(prior_cfg["conv_filter_size"]))
            dropout.append(float(prior_cfg.get("dropout_rate", 0.0)))
        return hidden_units, num_layers, conv_filter_size, dropout

    @staticmethod
    def normalize_state_dict_keys(state_dict: dict) -> dict:
        """!
        @brief Strip DataParallel `module.` prefixes when present.
        @param state_dict Raw state dict.
        @return Normalized state dict.
        """
        keys = list(state_dict.keys())
        if keys and all(k.startswith("module.") for k in keys):
            return {k[len("module."):]: v for k, v in state_dict.items()}
        return state_dict

    @staticmethod
    def infer_num_embeddings_from_state_dict(state_dict: dict, num_prior_levels: int) -> List[int]:
        """!
        @brief Infer codebook sizes from PixelCNN checkpoint tensors.
        @param state_dict PixelCNN state dict.
        @param num_prior_levels Number of prior levels.
        @return Per-level codebook sizes.
        """
        module_names = ["top_prior", "bottom_level"] if num_prior_levels == 2 else ["top_prior", "middle_level", "bottom_level"]
        result: List[int] = []
        for module_name in module_names:
            candidates = [f"{module_name}.embedding.weight", f"{module_name}.output_conv.3.weight"]
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

    @staticmethod
    def infer_two_level_conditioning_mode(config: dict, state_dict: dict) -> str:
        """!
        @brief Infer hierarchical PixelCNN conditioning mode from config/checkpoint.
        @param config PixelCNN config.
        @param state_dict PixelCNN state dict.
        @return Conditioning mode.
        """
        mode = config.get("model", {}).get("two_level_conditioning_mode", "deconv")
        first_key = "conditioning_stack.0.weight"
        if first_key in state_dict and state_dict[first_key].ndim == 4:
            kernel_h, kernel_w = state_dict[first_key].shape[2], state_dict[first_key].shape[3]
            if (kernel_h, kernel_w) == (4, 4):
                return "deconv"
            if (kernel_h, kernel_w) == (3, 3):
                return "conv"
        return mode
