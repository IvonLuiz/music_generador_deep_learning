import os
import pickle
import sys
import yaml
import argparse
from datetime import datetime
from typing import Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.jukebox_hierarchical_quantized_dataset import JukeboxHierarchicalQuantizedDataset
from modeling.torch.transformer_prior_conditioned import TransformerPriorConditioned
from utils import load_maestro, load_config
from train_scripts.jukebox_utils import load_jukebox_model, _parse_level
from test_scripts.test_transformer_prior import load_transformer_prior
from generation.soundgenerator import SoundGenerator
from processing.preprocess_audio import MIN_MAX_VALUES_SAVE_DIR, SAMPLE_RATE, HOP_LENGTH


def _prepare_min_max_values(min_max_values: object, count: int) -> list:
    if isinstance(min_max_values, dict):
        values = list(min_max_values.values())
    elif isinstance(min_max_values, list):
        values = min_max_values
    else:
        raise ValueError('min_max_values must be a dict or list')

    if not values:
        raise ValueError('min_max_values is empty')

    if len(values) >= count:
        return values[:count]

    repeats = (count + len(values) - 1) // len(values)
    tiled = (values * repeats)[:count]
    return tiled
def _save_decoded_spectrograms(specs: np.ndarray, save_dir: str) -> None:
    spec_dir = os.path.join(save_dir, 'spectrograms')
    os.makedirs(spec_dir, exist_ok=True)

    np.save(os.path.join(spec_dir, 'bottom_decoded_specs.npy'), specs)

    for i in range(specs.shape[0]):
        img = specs[i, :, :, 0]
        plt.figure(figsize=(6, 4))
        plt.imshow(img, origin='lower', aspect='auto')
        plt.colorbar()
        plt.title(f'Decoded Bottom Spectrogram {i}')
        plt.tight_layout()
        plt.savefig(os.path.join(spec_dir, f'bottom_spec_{i:03d}.png'), dpi=150)
        plt.close()
def _decode_bottom_indices(
    vqvae,
    indices: torch.Tensor,
    grid: Optional[list],
    device: torch.device,
) -> np.ndarray:
    if indices.ndim != 2:
        raise ValueError(f'Expected indices shape (B, T), got {tuple(indices.shape)}')
    if not (isinstance(grid, list) and len(grid) == 2):
        raise ValueError('Bottom grid is required to reshape indices into (H, W)')

    h, w = int(grid[0]), int(grid[1])
    if h * w != indices.shape[1]:
        raise ValueError(f'Grid {grid} does not match seq_len={indices.shape[1]}')

    idx_2d = indices.view(indices.shape[0], h, w).long().to(device)
    vqvae.eval()
    with torch.no_grad():
        emb = vqvae.vq.embedding[idx_2d]  # (B, H, W, D)
        z_q = emb.permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)
        x_hat = vqvae.decoder(z_q)
        if vqvae.activation_layer is not None:
            x_hat = vqvae.activation_layer(x_hat)

    return x_hat.detach().cpu().permute(0, 2, 3, 1).numpy()

bottom_transformer_prior_config_path = "models/transformer_prior/jukebox_maestro2011_bottom_transformer_prior/2026-03-23_17-55-25/config.yaml"
middle_transformer_prior_config_path = "models/transformer_prior/jukebox_maestro2011_middle_transformer_prior/2026-03-22_14-09-31/config.yaml"
top_transformer_prior_config_path = "models/transformer_prior/jukebox_maestro2011_top_transformer_prior/2026-03-22_13-55-04/config.yaml"

config_top = load_config(top_transformer_prior_config_path)
config_middle = load_config(middle_transformer_prior_config_path)
config_bottom = load_config(bottom_transformer_prior_config_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

top_prior, top_config, _ = load_transformer_prior('top', top_transformer_prior_config_path, device)
top_seq_len = int(top_config['model']['inferred_seq_lens']['top'])
top_grid = top_config['model'].get('inferred_grids', {}).get('top')

middle_prior, middle_config, _ = load_transformer_prior('middle', middle_transformer_prior_config_path, device)
middle_seq_len = int(middle_config['model']['inferred_seq_lens']['middle'])
middle_grid = middle_config['model'].get('inferred_grids', {}).get('middle')

bottom_prior, bottom_config, _ = load_transformer_prior('bottom', bottom_transformer_prior_config_path, device)
bottom_seq_len = int(bottom_config['model']['inferred_seq_lens']['bottom'])
bottom_grid = bottom_config['model'].get('inferred_grids', {}).get('bottom')

vqvae_bottom_decoder = load_jukebox_model(
    bottom_config['vqvae']['bottom_model_dir'],
    'bottom',
    device,
    bottom_config['vqvae']['weights_file'],
    )
vqvae_bottom_decoder.eval()


# Step 1: Top-Level Unrolling (The Composer)
## TODO: remove hardcoded generation parameters and make them configurable
temperature = 1.0
top_k = 100

## loop until generated amount of audio is reached, generating in blocks of top_seq_len and using the last part of the previous block as context for the next block
chunks_to_generate = 4  # Example value
curr_start_token = None
top_tokens_list = []
for chunk in range(chunks_to_generate):
    with torch.no_grad():
        top_tokens = top_prior.generate(
            batch_size=1,
            start_tokens=curr_start_token,
            upper_indices=None,
            seq_len=top_seq_len,
            temperature=temperature,  
            top_k=top_k,
            device=device,  
        ).cpu().numpy()

    ## overlap extract 
    ### take second half of the current tokens 
    ### and use it as the start tokens for the next level's generation
    overlap_len = len(top_grid) // 2 if top_grid else 0
    curr_start_token = top_tokens[:, -overlap_len:] if overlap_len > 0 else None
    curr_start_token = torch.from_numpy(curr_start_token).to(device) if curr_start_token is not None else None
    top_tokens_list.append(top_tokens)

# Step 2: Hierarchical Upsampling (The Performers)
curr_start_token = None
middle_tokens_list = []
for chunk in range(chunks_to_generate):
    curr_top_tokens = torch.from_numpy(top_tokens_list[chunk]).to(device)
    
    middle_tokens = middle_prior.generate(
        batch_size=1,
        start_tokens=curr_start_token,  # Use the last part of the previous block as context
        upper_indices=curr_top_tokens,
        seq_len=middle_seq_len,
        temperature=temperature,
        top_k=top_k,
        device=device,
    ).cpu().numpy()
    
    ## overlap extract for middle tokens to use as context for the next block
    overlap_len = len(middle_grid) // 2 if middle_grid else 0
    curr_start_token = middle_tokens[:, -overlap_len:] if overlap_len > 0 else None
    curr_start_token = torch.from_numpy(curr_start_token).to(device) if curr_start_token is not None else None
    middle_tokens_list.append(middle_tokens)
    
curr_start_token = None
bottom_tokens_list = []
for chunk in range(chunks_to_generate):
    curr_middle_tokens = torch.from_numpy(middle_tokens_list[chunk]).to(device)
    
    bottom_tokens = bottom_prior.generate(
        batch_size=1,
        start_tokens=curr_start_token,  # Use the last part of the previous block as context
        upper_indices=curr_middle_tokens,
        seq_len=bottom_seq_len,
        temperature=temperature,
        top_k=top_k,
        device=device,
    ).cpu().numpy()
    
    ## overlap extract for bottom tokens to use as context for the next block
    overlap_len = len(bottom_grid) // 2 if bottom_grid else 0
    curr_start_token = bottom_tokens[:, -overlap_len:] if overlap_len > 0 else None
    curr_start_token = torch.from_numpy(curr_start_token).to(device) if curr_start_token is not None else None
    bottom_tokens_list.append(bottom_tokens)

# At this point, bottom_tokens_list contains the generated token indices for each block, which can be decoded back
# into spectrograms using the VQ-VAE decoder. The overlapping regions between blocks can be blended together using
# a linear crossfade to create a seamless spectrogram, which can then be inverted back to audio using Griffin-Lim or a neural vocoder.

# Step 3: Spectrogram Crossfading (The Audio Engineer)
def linear_crossfading(spec_chunk_1, spec_chunk_2, overlap_len):
    """
    Linearly crossfade between two spectrogram chunks over the specified overlap length
    Formula: S_blended = (1-alfa) * S_1 + alfa * S_2
    
    @spec_chunk_1: Spectrogram chunk from the first block (shape: [channels, time_frames, freq_bins])
    @spec_chunk_2: Spectrogram chunk from the second block (shape: [channels, time_frames, freq_bins])
    @overlap_len: Number of time frames to use for the crossfade (must be > 0)
    @returns: Blended spectrogram chunk with the same shape as the input chunks
    """
    assert overlap_len > 0, "Overlap length must be greater than 0 for crossfading"
    
    # We reshape to (1, overlap_len, 1, 1) assuming shape is (Batch, time_frames, freq_bins, channels)
    fade_in = np.linspace(0, 1, num=overlap_len).reshape(1, overlap_len, 1, 1)
    fade_out = 1 - fade_in
    
    # Apply crossfade to the overlapping regions
    spec_chunk_1[:, -overlap_len:] *= fade_out
    spec_chunk_2[:, :overlap_len] *= fade_in
    # Combine the two chunks
    combined_chunk = np.concatenate([spec_chunk_1[:, :-overlap_len], spec_chunk_1[:, -overlap_len:] + spec_chunk_2[:, :overlap_len], spec_chunk_2[:, overlap_len:]], axis=1)
    return combined_chunk

reconstructed_spectrograms = []
for bottom_tokens in bottom_tokens_list:
    bottom_tokens_tensor = torch.from_numpy(bottom_tokens).to(device)
    with torch.no_grad():
        decoded_specs = _decode_bottom_indices(vqvae_bottom_decoder, bottom_tokens_tensor, bottom_grid, device)
    reconstructed_spectrograms.append(decoded_specs)


final_spectrogram = reconstructed_spectrograms[0].copy()
for i in range(1, len(reconstructed_spectrograms)):
    next_chunk = reconstructed_spectrograms[i].copy()
    overlap_len = min(final_spectrogram.shape[1], next_chunk.shape[1]) // 4
    final_spectrogram = linear_crossfading(final_spectrogram, next_chunk, overlap_len)

# Step 4: Spectrogram Inversion (The Mastering Engineer)
# The blended spectrograms can now be inverted back to audio using Griffin-Lim or a neural vocoder. This step is not implemented here, but libraries like librosa (for Griffin-Lim) or pretrained neural vocoders can be used for this purpose.

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_dir = os.path.join('samples', 'transformer_hierarchical_generated', current_time)
min_max_values_path = os.path.join(MIN_MAX_VALUES_SAVE_DIR, 'min_max_values.pkl')

if not os.path.exists(min_max_values_path):
    raise FileNotFoundError(f'min_max_values.pkl not found at {min_max_values_path}')

with open(min_max_values_path, 'rb') as f:
    min_max_values = pickle.load(f)

_save_decoded_spectrograms(final_spectrogram, save_dir)
min_max_list = _prepare_min_max_values(min_max_values, final_spectrogram.shape[0])

audio_method = 'griffinlim'  # or 'neural_vocoder'
import soundfile as sf
sound_generator = SoundGenerator(vqvae_bottom_decoder, hop_length=HOP_LENGTH)
audio_signals = sound_generator.convert_spectrograms_to_audio(
    final_spectrogram, min_max_list, method=audio_method
)

audio_dir = os.path.join(save_dir, 'audio')
os.makedirs(audio_dir, exist_ok=True)
for i, signal in enumerate(audio_signals):
    sf.write(os.path.join(audio_dir, f'sample_{i}.wav'), signal, SAMPLE_RATE)
print(f'Done! Saved generated samples to {save_dir}')
