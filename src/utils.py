import os
import numpy as np
from typing import Union
from tqdm import tqdm
import yaml
import torch

from modeling.torch.vq_vae import VQ_VAE
from modeling.torch.vq_vae_residual import VQ_VAE as VQ_VAE_Residual
from modeling.torch.vq_vae_hierarchical import VQ_VAE_Hierarchical
from modeling.torch.pixel_cnn import ConditionalGatedPixelCNN
from processing.preprocess_audio import TARGET_TIME_FRAMES


def load_maestro(path, target_time_frames=256, debug_print=False):
    # Check for cached dataset to speed up loading
    # We use the parent directory of the path if path is a directory, or just path
    cache_dir = path if os.path.isdir(path) else os.path.dirname(path)
    cache_path = os.path.join(cache_dir, "dataset_cache.npz")
    
    if os.path.exists(cache_path):
        print(f"Found cached dataset at {cache_path}. Loading...")
        try:
            data = np.load(cache_path)
            x_train = data['x_train']
            file_paths = data['file_paths']
            print(f"Loaded cached dataset shape: {x_train.shape}")
            return x_train, file_paths
        except Exception as e:
            print(f"Error loading cache: {e}. Reloading from source files.")

    x_train = []
    file_paths = []
    
    # First, collect all file paths to know the total for tqdm
    all_files = []
    print("Scanning files in:", path)
    for root, _, file_names in os.walk(path):
        for file_name in file_names:
            if file_name.endswith(".npy"):
                all_files.append(os.path.join(root, file_name))

    print(f"Found {len(all_files)} spectrograms. Loading...")

    for file_path in tqdm(all_files, desc="Loading spectrograms"):
        try:
            spectrogram = np.load(file_path)  # (n_bins, n_frames)
        except ValueError as ve:
            print(f"ValueError loading {file_path}: {ve}")
            continue
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
        
        # Debug: print shapes to see what we're getting
        if debug_print:
            print(f"Spectrogram {os.path.basename(file_path)} shape: {spectrogram.shape}")
            
        # Ensure consistent time dimension - crop or pad to exactly target time frames
        if spectrogram.shape[1] > target_time_frames:
            spectrogram = spectrogram[:, :target_time_frames]  # Crop to 256
        elif spectrogram.shape[1] < target_time_frames:
            # Pad to target time frames
            pad_width = target_time_frames - spectrogram.shape[1]
            spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
        
        x_train.append(spectrogram)
        file_paths.append(file_path)
    
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (N, target_frames, target_frames, 1) (N, 256, 256, 1)
    
    print(f"Final dataset shape: {x_train.shape}")
    
    # Save to cache for next time
    print(f"Saving dataset to cache {cache_path}...")
    np.savez(cache_path, x_train=x_train, file_paths=file_paths)
    print("Cache saved.")
    
    return x_train, file_paths


# Helper to locate the min/max entry for a spectrogram file
def find_min_max_for_path(fp, min_max_values, spectrograms_dir):
    bas = os.path.basename(fp)
    candidates = [
        fp,
        os.path.normpath(fp),
        os.path.abspath(fp),
        bas,
        os.path.join(spectrograms_dir, bas),
        os.path.abspath(os.path.join(spectrograms_dir, bas)),
    ]
    # also try with/without leading ./ or ../
    candidates += [c.replace('./', '') for c in list(candidates)]
    candidates += [c.replace('../', '') for c in list(candidates)]

    for c in candidates:
        if c in min_max_values:
            return min_max_values[c]
    # try matching by basename contained in any key
    for k, v in min_max_values.items():
        if bas == os.path.basename(k) or bas in k or os.path.basename(k) in bas:
            return v
    # not found
    return None

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def initialize_vqvae_model(config_or_path, device=torch.device('cpu')) -> Union[VQ_VAE, VQ_VAE_Residual]:
    """
    Initializes a VQ-VAE model from a configuration dictionary or path.
    
    Args:
        config_or_path (dict or str): Configuration dictionary or path to config.yaml.
        device (torch.device): Device to initialize the model onto.
        
    Returns:
        torch.nn.Module: The initialized VQ-VAE model (random weights).
    """
    if isinstance(config_or_path, str):
        config = load_config(config_or_path)
    else:
        config = config_or_path

    model_config = config['model'] if 'model' in config else config
    
    K = model_config['K']
    D = model_config['D']
    conv_filters = tuple(model_config['conv_filters'])
    conv_kernels = tuple(model_config['conv_kernels'])
    conv_strides = tuple([tuple(s) for s in model_config['conv_strides']])
    dropout_rate = model_config.get('dropout_rate', 0.0)
    use_residual = model_config.get('use_residual', False)
    
    if use_residual:
        print("Initializing Residual VQ-VAE...")
        model = VQ_VAE_Residual(
            input_shape=(256, TARGET_TIME_FRAMES, 1),
            conv_filters=conv_filters,
            conv_kernels=conv_kernels,
            conv_strides=conv_strides,
            embeddings_size=K,
            latent_space_dim=D,
            dropout_rate=dropout_rate
        )
    else:
        print("Initializing Standard VQ-VAE...")
        model = VQ_VAE(
            input_shape=(256, TARGET_TIME_FRAMES, 1),
            conv_filters=conv_filters,
            conv_kernels=conv_kernels,
            conv_strides=conv_strides,
            embeddings_size=K,
            latent_space_dim=D,
            dropout_rate=dropout_rate
        )
    
    model.to(device)
    return model

def initialize_vqvae_hierarchical_model(config_or_path, device=torch.device('cpu')) -> VQ_VAE_Hierarchical:
    """
    Initializes a Hierarchical VQ-VAE model from a configuration dictionary or path.
    
    Args:
        config_or_path (dict or str): Configuration dictionary or path to config.yaml.
        device (torch.device): Device to initialize the model onto.
    """
    
    if isinstance(config_or_path, str):
        config = load_config(config_or_path)
    else:
        config = config_or_path

    model_config = config['model'] if 'model' in config else config
    dim_bottom = model_config['dim_bottom']
    dim_top = model_config['dim_top']
    num_residual_layers = model_config['num_residual_layers']
    num_embeddings_top = model_config['num_embeddings_top']
    num_embeddings_bottom = model_config['num_embeddings_bottom']
    beta = model_config['beta']
    dropout_rate = model_config.get('dropout_rate', 0.0)
    
    model = VQ_VAE_Hierarchical(
        input_shape=(256, TARGET_TIME_FRAMES, 1),
        dim_bottom=dim_bottom,
        dim_top=dim_top,
        num_residual_layers=num_residual_layers,
        num_embeddings_top=num_embeddings_top,
        num_embeddings_bottom=num_embeddings_bottom,
        beta=beta,
        dropout_rate=dropout_rate
    )
    
    model.to(device)
    return model

def load_vqvae_model(model_path: str, device: torch.device, weights_file: str = None):
    """
    Loads a VQ-VAE model from a given path (initializes and loads weights).
    
    Args:
        model_path (str): Path to the model file (.pth) or the directory containing it.
                          If a directory is provided, it looks for 'model.pth' and 'config.yaml'.
                          If a file is provided, it looks for 'config.yaml' in the same directory.
        device (torch.device): Device to load the model onto.
        weights_file (str, optional): Specific weights file name to load if model_path is a directory.
                                      If None, tries to read 'weights_file_choice' from config, 
                                      defaulting to 'model.pth'.
    """
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.yaml")
        # Defer model_file determination
    else:
        config_path = os.path.join(os.path.dirname(model_path), "config.yaml")
        model_file = model_path
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    config = load_config(config_path)
    
    if os.path.isdir(model_path):
        if weights_file:
            model_file = os.path.join(model_path, weights_file)
        else:
            # Check config for preference, default to model.pth
            # Try 'testing' section first (new standard), then 'training' (legacy), then default
            choice = config.get('testing', {}).get('weights_file_choice')
            if not choice:
                choice = config.get('training', {}).get('weights_file_choice')
            
            if not choice:
                choice = "model.pth"
                
            model_file = os.path.join(model_path, choice)
    
    # Initialize model structure using the helper function
    model = initialize_vqvae_model(config, device)
        
    print(f"Loading VQ-VAE weights from {model_file}")
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    return model

def load_pixelcnn_model(model_path: str, device: torch.device, weights_file: str = None, num_embeddings: int = None):
    """
    Loads a PixelCNN model from a given path.
    
    Args:
        model_path (str): Path to the model file (.pth) or the directory containing it.
                          If a directory is provided, it looks for 'best_pixelcnn_model.pth' and 'config.yaml'.
        device (torch.device): Device to load the model onto.
        weights_file (str, optional): Specific weights file name.
        num_embeddings (int, optional): Number of embeddings (K). If provided, overrides config.
        
    Returns:
        torch.nn.Module: The loaded PixelCNN model.
    """
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.yaml")
        # Prefer best model
        if os.path.exists(os.path.join(model_path, "best_pixelcnn_model.pth")):
            model_file = os.path.join(model_path, "best_pixelcnn_model.pth")
        else:
            # Fallback to any .pth file or specific name
            model_file = os.path.join(model_path, "pixelcnn_model.pth")
    else:
        config_path = os.path.join(os.path.dirname(model_path), "config.yaml")
        model_file = model_path

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
