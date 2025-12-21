from datetime import datetime
import torch
import os
import yaml
import sys
import pickle

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling.torch.vq_vae_hierarchical import VQ_VAE_Hierarchical
from generation.generate import *
from utils import load_maestro, load_config, initialize_vqvae_hierarchical_model
from train_scripts.train_vqvae2_utils import train_vqvae_hierarquical
from processing.preprocess_audio import TARGET_TIME_FRAMES, MIN_MAX_VALUES_SAVE_DIR

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

if __name__ == "__main__":
    # Load configuration
    config_path = "./config/config_vqvae_hierarchical.yaml"
    config = load_config(config_path)
    print(f"Configuration loaded from {config_path}")
    print(config)

    # Optional: faster matmul on Ampere+ GPUs
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Capability:", torch.cuda.get_device_capability(0))
        print("CUDA memory allocated (MB):", round(torch.cuda.memory_allocated(0)/1024**2, 2))

    # Load configuration
    config_path = "./config/config_vqvae_hierarchical.yaml"
    config = load_config(config_path)

    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    epochs = config['training']['epochs']
    spectrograms_path = config['dataset']['processed_path']
    model_save_dir = config['training']['save_dir']
    model_name = config['model']['name']

    current_datetime = datetime.now()
    formatted_time = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    # structure: model_save_dir / formatted_time / model.pth
    run_dir = os.path.join(model_save_dir, formatted_time)
    os.makedirs(run_dir, exist_ok=True)

    model_file_path = os.path.join(run_dir, "model.pth")
    config_file_path = os.path.join(run_dir, "config.yaml")

    print(f"Training configuration loaded from {config_path}")
    print(f"Model will be saved to: {model_file_path}")

    # Save the config used for this training run immediately
    with open(config_file_path, 'w') as f:
        yaml.dump(config, f)
    
    # Load training data to compute variance
    x_train, file_paths = load_maestro(spectrograms_path, TARGET_TIME_FRAMES, debug_print=False)
    data_variance = np.var(x_train)

    # Load min_max_values
    min_max_values_path = os.path.join(MIN_MAX_VALUES_SAVE_DIR, "min_max_values.pkl")
    with open(min_max_values_path, "rb") as f:
        min_max_values = pickle.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vqvae_hierarquical_model = initialize_vqvae_hierarchical_model(config['model'], device)
    
    # Train the VQ-VAE Hierarchical model
    train_vqvae_hierarquical(
        model=vqvae_hierarquical_model,
        x_train=x_train,
        train_file_paths=file_paths,
        min_max_values=min_max_values,
        device=device,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        save_path=model_file_path,
        data_variance=data_variance
    )
    print("Training completed.")