# %%
import torch

import sys
sys.path.insert(1, '../src')

from modeling.torch.vq_vae import VQ_VAE
from modeling.torch.train_vq import *
from generate import *

# %%
import os
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



# %%
SPECTROGRAMS_SAVE_DIR = "data/processed/maestro_spectrograms/"
SPECTROGRAMS_PATH = "data/processed/maestro_spectrograms/"
MODEL_PATH = "../model/vq_vae_maestro2011/model.pth"

# %%
MIN_MAX_VALUES_SAVE_DIR = "data/raw/maestro-v3.0.0/2011/min_max_values.pkl"
MIN_MAX_VALUES_PATH = "data/raw/maestro-v3.0.0/2011/min_max_values.pkl"

SAVE_DIR_ORIGINAL = "samples/vq_vae_maestro2011/original/"
SAVE_DIR_GENERATED = "samples/vq_vae_maestro2011/generated/"

# %%
if os.path.exists(SAVE_DIR_ORIGINAL) is False:
    os.makedirs(SAVE_DIR_ORIGINAL)
if os.path.exists(SAVE_DIR_GENERATED) is False:
    os.makedirs(SAVE_DIR_GENERATED)

# %%
def load_maestro(path):
    x_train = []
    file_paths = []
    
    for root, _, file_names in os.walk(path):
        for file_name in file_names:
            if file_name.endswith(".npy"):
                file_path = os.path.join(root, file_name)
                spectrogram = np.load(file_path) # (n_bins, n_frames, 1)
                x_train.append(spectrogram)
                file_paths.append(file_path)
    
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1)

    return x_train, file_paths

# %%
x_train, _ = load_maestro(SPECTROGRAMS_PATH)

# %%
print(x_train.shape)

# %%
data_variance = np.var(x_train / 255.0)

# %%
VQVAE = VQ_VAE(
    input_shape=(256, x_train.shape[2], 1),
    conv_filters=(256, 128, 64, 32),
    conv_kernels=(3, 3, 3, 3),
    conv_strides=(2, 2, 2, (2, 1)),
    #data_variance=data_variance,
    embeddings_size=256,    # K
    latent_space_dim=256    # D
)


# %%
# The model was saved during training; keep a reference to the path
print(f"Model path: {MODEL_PATH}")

# %%
# Load checkpoint (optional)
ckpt = torch.load(MODEL_PATH, map_location='cuda')
VQVAE.load_state_dict(ckpt['model_state'])
VQVAE.eval()

# %% [markdown]
# ## Generate
# 

# %%
sound_generator = SoundGenerator(VQVAE, 256)

# %%
# Load spectrograms + min max values
with open(MIN_MAX_VALUES_PATH, "rb") as f:
    min_max_values = pickle.load(f)
specs, file_paths = load_maestro(SPECTROGRAMS_PATH)


# %%
# Sample spectrograms + min max values (robust matching to preprocessing keys)
import os
import pickle
import numpy as np

# Paths in this notebook
# SPECTROGRAMS_PATH and MIN_MAX_VALUES_SAVE_DIR are set in earlier cells
print('SPECTROGRAMS_PATH =', SPECTROGRAMS_PATH)
print('MIN_MAX_VALUES_SAVE_DIR =', MIN_MAX_VALUES_SAVE_DIR)

with open(MIN_MAX_VALUES_SAVE_DIR, 'rb') as f:
    min_max_values = pickle.load(f)

specs, file_paths = load_maestro(SPECTROGRAMS_PATH)
print('Found', len(specs), 'spectrogram files')

# Helper to locate the min/max entry for a spectrogram file
def find_min_max_for_path(fp, min_max_values, spectrograms_dir=SPECTROGRAMS_PATH):
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

# Sample some spectrograms
num_samples = min(5, len(specs))
indexes = np.random.choice(range(len(specs)), num_samples, replace=False)
sampled_specs = specs[indexes]
sampled_paths = [file_paths[i] for i in indexes]

sampled_min_max_values = []
for p in sampled_paths:
    mm = find_min_max_for_path(p, min_max_values)
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
originals = sound_generator.convert_spectrograms_to_audio(sampled_specs, sampled_min_max_values)

# Save outputs
save_signals(signals, SAVE_DIR_GENERATED)
save_signals(originals, SAVE_DIR_ORIGINAL)
print('Saved generated ->', SAVE_DIR_GENERATED)
print('Saved originals ->', SAVE_DIR_ORIGINAL)

# %%
# Generate audio for sampled spectrograms using the trained torch model
signals, _ = sound_generator.generate(sampled_specs, sampled_min_max_values)

# Listen to the first generated sample (requires IPython.display)
import IPython.display as ipd
if len(signals) > 0:
    ipd.display(ipd.Audio(signals[0], rate=22050))

# Optionally, listen to the original for comparison
original_signals = sound_generator.convert_spectrograms_to_audio(sampled_specs, sampled_min_max_values)
if len(original_signals) > 0:
    ipd.display(ipd.Audio(original_signals[0], rate=22050))

# %%
save_signals(signals, SAVE_DIR_GENERATED)
save_signals(original_signals, SAVE_DIR_ORIGINAL)


