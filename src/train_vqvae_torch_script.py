from datetime import datetime
import torch
import os
import yaml

from modeling.torch.vq_vae import VQ_VAE
from modeling.torch.vq_vae_residual import VQ_VAE as VQ_VAE_Residual
from modeling.torch.train_vq import *
from generation.generate import *
from utils import load_maestro, load_config
from processing.preprocess_audio import TARGET_TIME_FRAMES

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Load configuration
CONFIG_PATH = "./config/config.yaml"
config = load_config(CONFIG_PATH)

# Optional: faster matmul on Ampere+ GPUs
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
    print("CUDA memory allocated (MB):", round(torch.cuda.memory_allocated(0)/1024**2, 2))

# Model parameters from config
K = config['model']['K']
D = config['model']['D']
conv_filters = tuple(config['model']['conv_filters'])
conv_kernels = tuple(config['model']['conv_kernels'])
conv_strides = tuple([tuple(s) for s in config['model']['conv_strides']])
dropout_rate = config['model']['dropout_rate']
use_residual = config['model'].get('use_residual', False)

# Training parameters from config
BATCH_SIZE = config['training']['batch_size']
LEARNING_RATE = config['training']['learning_rate']
EPOCHS = config['training']['epochs']

# Paths from config
SPECTROGRAMS_PATH = config['dataset']['processed_path']
MODEL_SAVE_DIR = config['training']['save_dir']
MODEL_NAME = config['model']['name']

current_datetime = datetime.now()
formatted_time = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

# Construct model path with hyperparameters for uniqueness
MODEL_PATH = os.path.join(
    MODEL_SAVE_DIR, 
    f"{MODEL_NAME}_K_{K}_D_{D}_filters_{'_'.join(map(str, conv_filters))}"
)
if use_residual:
    MODEL_PATH += "_residual"

MODEL_FILE_PATH = os.path.join(MODEL_PATH, f"{MODEL_NAME}_model.pth")

print(f"Training configuration loaded from {CONFIG_PATH}")
print(f"Model will be saved to: {MODEL_FILE_PATH}")

x_train, _ = load_maestro(SPECTROGRAMS_PATH, TARGET_TIME_FRAMES, debug_print=False)
print("Input shape: ", x_train.shape)
print("Data range:", x_train.min(), "to", x_train.max())
print("Data samples length:", x_train.shape[0])

# If variance is too small, the loss term (MSE / 2*var) becomes huge, causing exploding gradients.
data_variance = np.var(x_train) # data cines normalized from load_maestro
print(f"Data variance: {data_variance}")

# Get the actual time dimension from your data
time_frames = x_train.shape[2]
print(f"Time frames detected: {time_frames}")

# Define VQ-VAE model with actual dimensions
if use_residual:
    print("Initializing Residual VQ-VAE...")
    # Note: VQ_VAE_Residual might not support dropout_rate in __init__ if not updated
    # Assuming it has similar signature or we need to check vq_vae_residual.py
    # Based on previous context, vq_vae_residual.py didn't have dropout_rate in __init__
    # Let's assume we want to use it if available, or fallback.
    # Ideally, we should update vq_vae_residual.py to match vq_vae.py features.
    # For now, let's instantiate without dropout if it fails, or update the file first.
    # Actually, let's update vq_vae_residual.py to support dropout first to be safe.
    VQVAE = VQ_VAE_Residual(
        input_shape=(256, time_frames, 1),
        conv_filters=conv_filters,
        conv_kernels=conv_kernels,
        conv_strides=conv_strides,
        embeddings_size=K,
        latent_space_dim=D,
        dropout_rate=dropout_rate
    )
else:
    print("Initializing Standard VQ-VAE...")
    VQVAE = VQ_VAE(
        input_shape=(256, time_frames, 1),
        conv_filters=conv_filters,
        conv_kernels=conv_kernels,
        conv_strides=conv_strides,
        embeddings_size=K,
        latent_space_dim=D,
        dropout_rate=dropout_rate
    )

# Train the model using the train_model function (with AMP to save memory)
# We save to a timestamped subdirectory to keep history
timestamped_model_path = os.path.join(MODEL_PATH, formatted_time, "model.pth")

# Also save the config used for this training run
os.makedirs(os.path.dirname(timestamped_model_path), exist_ok=True)
with open(os.path.join(os.path.dirname(timestamped_model_path), "config.yaml"), 'w') as f:
    yaml.dump(config, f)

train_model(
    VQVAE,
    x_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    data_variance=data_variance,
    save_path=timestamped_model_path,
    amp=True,
    grad_accum_steps=1,
    max_grad_norm=1.0, # Add gradient clipping to prevent explosion
    model_config=config['model'] # Save model config inside checkpoint
)
print("Model training complete. Model saved to:", timestamped_model_path)

# Update a "latest" symlink or file to point to this model for easy testing
latest_model_pointer = os.path.join(MODEL_SAVE_DIR, "latest_model_path.txt")
with open(latest_model_pointer, 'w') as f:
    f.write(timestamped_model_path)
print(f"Updated latest model pointer at {latest_model_pointer}")