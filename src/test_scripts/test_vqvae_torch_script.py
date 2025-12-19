from datetime import datetime
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_scripts.train_vq_utils import *
from generation.generate import *
from utils import load_maestro, find_min_max_for_path, load_config, load_vqvae_model
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

## Prepare paths and directories
current_datetime = datetime.now()
formatted_time = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

# Construct the full path to the specific run directory
run_dir_name = str(config['testing']['specific_run_dir'])
base_save_dir = config['training']['save_dir']
run_dir = os.path.join(base_save_dir, run_dir_name)

# Load run-specific config to ensure dataset params match
run_config_path = os.path.join(run_dir, "config.yaml")
if os.path.exists(run_config_path):
    print(f"Loading run-specific config from {run_config_path}")
    run_config = load_config(run_config_path)
    config.update(run_config)
else:
    print(f"Warning: Run-specific config not found at {run_config_path}. Using global config.")

weights_file = config['testing']['weights_file_choice']

spectrograms_path = config['dataset']['processed_path']
min_max_values_file_path = MIN_MAX_VALUES_SAVE_DIR + "min_max_values.pkl"
hop_length = config['dataset']['hop_length']

SAVE_DIR = f"samples/{config['model']['name']}/{formatted_time}/"

print('spectrograms_path =', spectrograms_path)
print('min_max_values_file_path =', min_max_values_file_path)

with open(min_max_values_file_path, 'rb') as f:
    min_max_values = pickle.load(f)

specs, file_paths = load_maestro(spectrograms_path, TARGET_TIME_FRAMES)
print(specs.shape)
print("Data range:", specs.min(), "to", specs.max())
data_variance = np.var(specs)

## Load trained model
print(f"Model path: {run_dir}")
print(f"Weights file: {weights_file}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vqvae_model = load_vqvae_model(run_dir, device, weights_file=weights_file)

## Generate
sound_generator = SoundGenerator(vqvae_model, hop_length=hop_length)

# Sample some spectrograms
np.random.seed(42)
num_samples = min(5, len(specs))
indexes = np.random.choice(range(len(specs)), num_samples, replace=False)
sampled_specs = specs[indexes]
sampled_paths = [file_paths[i] for i in indexes]

sampled_min_max_values = []
for p in sampled_paths:
    mm = find_min_max_for_path(p, min_max_values, spectrograms_path)
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
