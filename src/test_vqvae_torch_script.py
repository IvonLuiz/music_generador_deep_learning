from datetime import datetime
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import yaml

from modeling.torch.vq_vae import VQ_VAE
from modeling.torch.vq_vae_residual import VQ_VAE as VQ_VAE_Residual
from modeling.torch.train_vq import *
from generation.generate import *
from utils import load_maestro, find_min_max_for_path, load_config
from processing.preprocess_audio import TARGET_TIME_FRAMES, MIN_MAX_VALUES_SAVE_DIR

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Load configuration
CONFIG_PATH = "./config/config_vqvae.yaml"
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

current_datetime = datetime.now()
formatted_time = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

# Try to load the latest trained model path
LATEST_MODEL_POINTER = os.path.join(config['training']['save_dir'], "latest_model_path.txt")
if os.path.exists(LATEST_MODEL_POINTER):
    with open(LATEST_MODEL_POINTER, 'r') as f:
        MODEL_PATH = f.read().strip()
    print(f"Using latest model from: {MODEL_PATH}")
else:
    print(f"Latest model pointer not found: {LATEST_MODEL_POINTER}.")
    exit(1)

# Load the config specific to this model run if it exists
model_run_config_path = os.path.join(os.path.dirname(MODEL_PATH), "config.yaml")
if os.path.exists(model_run_config_path):
    print(f"Loading run-specific config from {model_run_config_path}")
    run_config = load_config(model_run_config_path)
    # Update model params from the run config to ensure architecture matches
    config['model'] = run_config['model']

# Model parameters
K = config['model']['K']
D = config['model']['D']
conv_filters = tuple(config['model']['conv_filters'])
conv_kernels = tuple(config['model']['conv_kernels'])
conv_strides = tuple([tuple(s) for s in config['model']['conv_strides']])
dropout_rate = config['model'].get('dropout_rate', 0.0)
use_residual = config['model'].get('use_residual', False)

SPECTROGRAMS_PATH = config['dataset']['processed_path']
MIN_MAX_VALUES_FILE_PATH = MIN_MAX_VALUES_SAVE_DIR + "min_max_values.pkl"
HOP_LENGTH = config['dataset']['hop_length']

SAVE_DIR = f"samples/{config['model']['name']}/{formatted_time}/"

print('SPECTROGRAMS_PATH =', SPECTROGRAMS_PATH)
print('MIN_MAX_VALUES_FILE_PATH =', MIN_MAX_VALUES_FILE_PATH)

with open(MIN_MAX_VALUES_FILE_PATH, 'rb') as f:
    min_max_values = pickle.load(f)

specs, file_paths = load_maestro(SPECTROGRAMS_PATH, TARGET_TIME_FRAMES)
print(specs.shape)
print("Data range:", specs.min(), "to", specs.max())
data_variance = np.var(specs)

if use_residual:
    print("Initializing Residual VQ-VAE...")
    VQVAE = VQ_VAE_Residual(
        input_shape=(256, specs.shape[2], 1),
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
        input_shape=(256, specs.shape[2], 1),
        conv_filters=conv_filters,
        conv_kernels=conv_kernels,
        conv_strides=conv_strides,
        embeddings_size=K,
        latent_space_dim=D,
        dropout_rate=dropout_rate
    )

# Load trained model
print(f"Model path: {MODEL_PATH}")
ckpt = torch.load(MODEL_PATH, map_location='cuda')
VQVAE.load_state_dict(ckpt['model_state'])
VQVAE.eval()

## Generate
sound_generator = SoundGenerator(VQVAE, hop_length=HOP_LENGTH)


# Sample some spectrograms
np.random.seed(42)
num_samples = min(5, len(specs))
indexes = np.random.choice(range(len(specs)), num_samples, replace=False)
sampled_specs = specs[indexes]
sampled_paths = [file_paths[i] for i in indexes]

sampled_min_max_values = []
for p in sampled_paths:
    mm = find_min_max_for_path(p, min_max_values, SPECTROGRAMS_PATH)
    if mm is None:
        print(f"Warning: no min/max for {p}; using default")
        mm = {"min": -80.0, "max": 0.0}
    sampled_min_max_values.append(mm)

print('Sampled files:')
for p, mm in zip(sampled_paths, sampled_min_max_values):
    print(' -', p, '->', 'min' in mm and mm['min'], 'max' in mm and mm['max'])

# Generate with the trained PyTorch model
signals, latents = sound_generator.generate(sampled_specs, sampled_min_max_values)
# Also convert originals (denorm + istft) and play
originals = sound_generator.convert_spectrograms_to_audio(sampled_specs, sampled_min_max_values, method="griffinlim") # griffinlim or istft

save_multiple_signals({'generated': signals, 'original': originals}, SAVE_DIR)
save_spectrogram_comparisons(sampled_specs, sampled_min_max_values, sound_generator,
                             save_dir=f"{SAVE_DIR}/spectrograms/")
