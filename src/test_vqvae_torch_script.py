import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime

from modeling.torch.vq_vae import VQ_VAE
from modeling.torch.train_vq import *
from generation.generate import *
from utils import load_maestro, find_min_max_for_path
from processing.preprocess_audio import TARGET_TIME_FRAMES, MIN_MAX_VALUES_SAVE_DIR

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

current_datetime = datetime.now()
formatted_time = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

# Model parameters
K = 1024  # Number of embeddings
D = 512  # Latent space dimension
SPECTROGRAMS_PATH = "data/processed/maestro_spectrograms_test/"
MODEL_PATH = f"./models/vq_vae_maestro2011_K_{K}_D_{D}/vq_vae_maestro2011_model.pth"

MIN_MAX_VALUES_FILE_PATH = MIN_MAX_VALUES_SAVE_DIR + "min_max_values.pkl"

HOP_LENGTH = 256 # from preprocessing audio

SAVE_DIR = f"samples/vq_vae_maestro2011/{formatted_time}/"

print('SPECTROGRAMS_PATH =', SPECTROGRAMS_PATH)
print('MIN_MAX_VALUES_FILE_PATH =', MIN_MAX_VALUES_FILE_PATH)

with open(MIN_MAX_VALUES_FILE_PATH, 'rb') as f:
    min_max_values = pickle.load(f)

specs, file_paths = load_maestro(SPECTROGRAMS_PATH, TARGET_TIME_FRAMES)
print(specs.shape)
print("Data range:", specs.min(), "to", specs.max())
data_variance = np.var(specs)

VQVAE = VQ_VAE(
    input_shape=(256, specs.shape[2], 1),
    conv_filters=(32, 64, 128, 256),
    conv_kernels=(3, 3, 3, 3),
    conv_strides=((2, 2), (2, 2), (2, 2), (2, 1)),
    embeddings_size=K,    # K
    latent_space_dim=D,    # D
    dropout_rate=0.1
)

# Load trained model
print(f"Model path: {MODEL_PATH}")
ckpt = torch.load(MODEL_PATH, map_location='cuda')
VQVAE.load_state_dict(ckpt['model_state'])
VQVAE.eval()

## Generate
sound_generator = SoundGenerator(VQVAE, hop_length=HOP_LENGTH)


# Sample some spectrograms
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
