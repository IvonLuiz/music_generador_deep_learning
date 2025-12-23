from datetime import datetime
import torch
import os
import yaml
import sys

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_scripts.train_vq_utils import *
from generation.generate import *
from utils import load_maestro, load_config, initialize_vqvae_model
from processing.preprocess_audio import TARGET_TIME_FRAMES
from datasets.spectrogram_dataset import MmapSpectrogramDataset
import gc
import numpy as np

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
config_path = "./config/config_vqvae.yaml"
config = load_config(config_path)

# Training parameters from config
batch_size = config['training']['batch_size']
learning_rate = config['training']['learning_rate']
epochs = config['training']['epochs']

# Paths from config
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

# Ensure testing configuration is present for future loading
if 'testing' not in config:
    config['testing'] = {}
if 'weights_file_choice' not in config['testing']:
    config['testing']['weights_file_choice'] = 'model.pth'

# Save the config used for this training run immediately
with open(config_file_path, 'w') as f:
    yaml.dump(config, f)

x_all, _ = load_maestro(spectrograms_path, TARGET_TIME_FRAMES, debug_print=False)
print("Input shape: ", x_all.shape)
print("Data range:", x_all.min(), "to", x_all.max())
print("Data samples length:", x_all.shape[0])

# If variance is too small, the loss term (MSE / 2*var) becomes huge, causing exploding gradients.
data_variance = np.var(x_all) # data cines normalized from load_maestro
print(f"Data variance: {data_variance}")

# Split into train/val
validation_split = config['training'].get('validation_split', 0.0)
if validation_split > 0:
    num_samples = len(x_all)
    num_val = int(num_samples * validation_split)
    num_train = num_samples - num_val
    
    # Shuffle indices
    indices = np.random.permutation(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    # Use MmapSpectrogramDataset to avoid loading data into RAM
    x_train = MmapSpectrogramDataset(x_all, train_indices)
    x_val = MmapSpectrogramDataset(x_all, val_indices)
    
    print(f"Data split: {len(x_train)} training, {len(x_val)} validation samples.")
    gc.collect()
else:
    x_train = MmapSpectrogramDataset(x_all)
    x_val = None
    print(f"Using all {len(x_train)} samples for training.")

# Get the actual time dimension from your data
time_frames = x_all.shape[2]
print(f"Time frames detected: {time_frames}")

# Define VQ-VAE model using the utility function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vqvae_model = initialize_vqvae_model(config, device)

# Train the model using the train_model function (with AMP to save memory)
train_model(
    vqvae_model,
    x_train,
    batch_size=batch_size,
    epochs=epochs,
    learning_rate=learning_rate,
    data_variance=data_variance,
    save_path=model_file_path,
    amp=True,
    grad_accum_steps=1,
    max_grad_norm=1.0, # Add gradient clipping to prevent explosion
    model_config=config['model'], # Save model config inside checkpoint
    x_val=x_val
)
print("Model training complete. Model saved to:", model_file_path)
