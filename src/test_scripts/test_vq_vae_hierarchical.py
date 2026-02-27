
import argparse
import sys
import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_config, load_vqvae_hierarchical_model_wrapper, load_maestro, find_min_max_for_path
from generation.soundgenerator import SoundGenerator
from generation.generate import save_multiple_signals, save_spectrogram_comparisons
from processing.preprocess_audio import TARGET_TIME_FRAMES, MIN_MAX_VALUES_SAVE_DIR

def test_vqvae_hierarchical(model_path, n_samples=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading Hierarchical VQ-VAE from {model_path}")
    model = load_vqvae_hierarchical_model_wrapper(model_path, device)
    
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.yaml")
    else:
        config_path = os.path.join(os.path.dirname(model_path), "config.yaml")
        
    config = load_config(config_path)
    
    # Dataset config
    dataset_config = config.get('dataset', {})
    spectrograms_path = dataset_config.get('processed_path', "./data/processed/maestro_spectrograms/")
    hop_length = dataset_config.get('hop_length', 256)
    
    # Load Data
    min_max_values_file_path = os.path.join(MIN_MAX_VALUES_SAVE_DIR, "min_max_values.pkl")
    if not os.path.exists(min_max_values_file_path):
        # Fallback location?
        min_max_values_file_path = "./data/min_max_values.pkl"
        
    print(f"Loading data from {spectrograms_path}")
    print(f"Min-max values: {min_max_values_file_path}")
    
    try:
        with open(min_max_values_file_path, 'rb') as f:
            min_max_values = pickle.load(f)
    except FileNotFoundError:
        print("Warning: min_max_values.pkl not found. Denormalization might be incorrect.")
        min_max_values = {}

    specs, file_paths = load_maestro(spectrograms_path, TARGET_TIME_FRAMES)
    print(f"Loaded {len(specs)} spectrograms. Shape: {specs.shape}")
    
    # Sample
    n_samples = min(n_samples, len(specs))
    indices = np.random.choice(len(specs), n_samples, replace=False)
    sampled_specs = specs[indices] # (N, H, W, 1)
    sampled_paths = [file_paths[i] for i in indices]
    
    sampled_min_max_values = []
    for p in sampled_paths:
        mm = find_min_max_for_path(p, min_max_values, spectrograms_path)
        if mm is None:
            # Try basename match if full path match fails
            if p in min_max_values:
                 mm = min_max_values[p]
            else:
                 mm = {"min": -80.0, "max": 0.0}
        sampled_min_max_values.append(mm)
        
    # Reconstruct
    print("Reconstructing samples...")
    # Prepare input tensor
    x_in = torch.from_numpy(sampled_specs).permute(0, 3, 1, 2).float().to(device) # (N, 1, H, W)
    
    model.eval()
    with torch.no_grad():
        x_recon, _, _ = model(x_in)
        # Assuming model outputs sigmoid (0-1), which matches input range
        
    # Convert to numpy (N, H, W, 1)
    gen_specs = x_recon.permute(0, 2, 3, 1).cpu().numpy()
    
    # Convert to Audio
    sound_generator = SoundGenerator(model, hop_length=hop_length)
    
    print("Converting to audio...")
    recon_signals = sound_generator.convert_spectrograms_to_audio(gen_specs, sampled_min_max_values)
    original_signals = sound_generator.convert_spectrograms_to_audio(sampled_specs, sampled_min_max_values)
    
    # Save
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"samples/vq_vae_hierarchical_test/{timestamp}/"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Saving samples to {save_dir}")
    save_multiple_signals({'reconstructed': recon_signals, 'original': original_signals}, save_dir)
    
    # Save Spectrogram plots
    spec_dir = os.path.join(save_dir, "spectrograms")
    os.makedirs(spec_dir, exist_ok=True)
    
    for i in range(n_samples):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original
        axes[0].imshow(sampled_specs[i, :, :, 0], origin='lower', aspect='auto')
        axes[0].set_title(f"Original {i}")
        
        # Reconstructed
        axes[1].imshow(gen_specs[i, :, :, 0], origin='lower', aspect='auto')
        axes[1].set_title(f"Reconstructed {i}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(spec_dir, f"comparison_{i}.png"))
        plt.close()
        
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to Hierarchical VQ-VAE model directory or .pth file")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples to reconstruct")
    
    args = parser.parse_args()
    
    test_vqvae_hierarchical(args.model_path, args.n_samples)
