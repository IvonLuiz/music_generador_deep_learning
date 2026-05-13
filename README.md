# Backing-track AI Generator

Music generation project based on discrete latent models for spectrograms:

- VQ-VAE (PyTorch)
- VQ-VAE with residual stack
- Hierarchical VQ-VAE (VQ-VAE-2 style)
- Jukebox-style multi-level VQ-VAE training (Bottom / Middle / Top)
- PixelCNN prior (single latent level)
- Hierarchical Conditional PixelCNN prior (Top + Bottom)

The project currently trains on MAESTRO spectrograms and reconstructs/generates audio using Griffin-Lim.

## Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Current Implemented Pipelines

### 1) VQ-VAE (single level)

- Main model: `src/modeling/torch/vq_vae.py`
- Residual variant: `src/modeling/torch/vq_vae_residual.py`
- Config: `config/config_vqvae.yaml`
- Trainer: `src/train_scripts/train_vqvae_torch_script.py`

Run:

```bash
python src/train_scripts/train_vqvae_torch_script.py
```

### 2) Hierarchical VQ-VAE (VQ-VAE-2)

- Model: `src/modeling/torch/vq_vae_hierarchical.py`
- Config: `config/config_vqvae_hierarchical.yaml`
- Trainer: `src/train_scripts/train_vqvae_hierarchical.py`

Run:

```bash
python src/train_scripts/train_vqvae_hierarchical.py
```

### 3) Jukebox-style VQ-VAE levels (Bottom / Middle / Top)

- Model: `src/modeling/torch/jukebox_vq_vae.py`
- Config: `config/config_jukebox.yaml`
- Single-level trainer: `src/train_scripts/train_vqvae_jukebox.py`
- Sequential trainer (all levels): `src/train_scripts/train_vqvae_jukebox_all_levels.sh`

Run one level:

```bash
python src/train_scripts/train_vqvae_jukebox.py --level bottom
python src/train_scripts/train_vqvae_jukebox.py --level middle
python src/train_scripts/train_vqvae_jukebox.py --level top
```

Run all levels in sequence:

```bash
bash src/train_scripts/train_vqvae_jukebox_all_levels.sh
```

### 4) PixelCNN prior (single level)

- Model: `src/modeling/torch/pixel_cnn.py`
- Config: `config/config_pixelcnn.yaml`
- Trainer: `src/train_scripts/train_pixel_cnn.py`

Run:

```bash
python src/train_scripts/train_pixel_cnn.py
```

This script loads the latest VQ-VAE run from the VQ-VAE save directory and trains an autoregressive prior over codebook indices.

### 5) Hierarchical PixelCNN prior (Top + Bottom)

- Model: `src/modeling/torch/pixel_cnn_hierarchical.py`
- Config: `config/config_pixelcnn_hierarchical.yaml`
- Trainer: `src/train_scripts/train_pixel_cnn_hierarchical.py`

Run with default config:

```bash
python src/train_scripts/train_pixel_cnn_hierarchical.py
```

Run with explicit arguments:

```bash
python src/train_scripts/train_pixel_cnn_hierarchical.py \
  --config ./config/config_pixelcnn_hierarchical.yaml \
  --vqvae ./models/vq_vae_hierarchical/<run_dir>/best_model.pth
```

### 6) Jukebox 3-level PixelCNN priors (Top + Middle + Bottom)

- Config: `config/config_pixelcnn_jukebox_hierarchical.yaml`
- Trainer: `src/train_scripts/train_pixel_cnn_jukebox_hierarchical.py`

Run:

```bash
python src/train_scripts/train_pixel_cnn_jukebox_hierarchical.py \
   --config ./config/config_pixelcnn_jukebox_hierarchical.yaml
```

This follows a Jukebox-style cascade during training:

- top prior: unconditional on top codes
- middle prior: conditioned on top codes
- bottom prior: conditioned on middle codes

## Training Order (Recommended)

1. Train a VQ model:
   - `train_vqvae_torch_script.py` **or** `train_vqvae_hierarchical.py` **or** Jukebox level training.
2. Train the corresponding prior:
   - `train_pixel_cnn.py` for single-level VQ-VAE.
   - `train_pixel_cnn_hierarchical.py` for hierarchical VQ-VAE-2 latents.
3. Generate samples and convert spectrograms to audio through `src/generation/soundgenerator.py`.

## Config Files

- `config/config_vqvae.yaml`: single-level VQ-VAE settings.
- `config/config_vqvae_hierarchical.yaml`: hierarchical VQ-VAE-2 settings.
- `config/config_jukebox.yaml`: Jukebox profile levels (`bottom`, `middle`, `top`).
- `config/config_pixelcnn.yaml`: single-level PixelCNN prior settings.
- `config/config_pixelcnn_hierarchical.yaml`: hierarchical PixelCNN prior settings.

## Tests

### Jukebox priors
To test only the bottom prior generation, we have a script that uses middle tokens directly from the VQ-VAE quantization of a real song, and then samples the bottom prior conditioned on the middle codes:

```bash
python3 src/test_scripts/test_bottom_prior_conditioned.py \
  --bottom_prior ./models/transformer_prior/jukebox_maestro_bottom_transformer_prior/2026-05-05_23-31-48/ \
  --data_root /home/ivon/code/datasets/processed/maestro_quantized_dataset_overlap50/ \
  --bottom_vqvae models/jukebox_vq_vae/jukebox_vqvae_maestro_bottom/2026-04-26_22-26-06 \
  --file MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--1.wav \
  --full_length
```

To do the same but add the middle and test it's quality, we have a script that the top tokens directly from the VQ-VAE quantization of a real song, and then samples the middle and bottom in this order conditioned on the top codes:

```bash
python3 src/test_scripts/test_middle_bottom_prior_conditioned.py  \
  --bottom_prior ./models/transformer_prior/jukebox_maestro_bottom_transformer_prior/2026-05-05_23-31-48/ --middle_prior ./models/transformer_prior/jukebox_maestro_middle_transformer_prior/2026-05-09_02-11-32/  \
  --data_root /home/ivon/code/datasets/processed/maestro_quantized_dataset_overlap50/ \
  --bottom_vqvae models/jukebox_vq_vae/jukebox_vqvae_maestro_bottom/2026-04-26_22-26-06 \
  --file MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--1.wav \
  --full_length
```

To test the full cascade of top + middle + bottom priors, we have a script that samples the top prior unconditionally, then samples the middle conditioned on the top, and finally samples the bottom conditioned on the middle:

```bash
python3 src/test_scripts/test_transformer_prior.py \
  --top_prior models/transformer_prior/jukebox_maestro_top_transformer_prior/2026-05-04_09-35-13 \
  --middle_prior models/transformer_prior/jukebox_maestro_middle_transformer_prior/2026-05-09_02-11-32/ \
  --bottom_prior models/transformer_prior/jukebox_maestro_bottom_transformer_prior/2026-05-05_23-31-48/ \
  --full_length \
  --full_length_until bottom \
  --decode_level bottom \
  --full_length_overlap_fraction 0.5 \
  --timing_duration_seconds 240
```

## Project Status

- [x] VAE (baseline)
- [x] VQ-VAE (PyTorch)
- [x] VQ-VAE residual variant
- [x] Hierarchical VQ-VAE (VQ-VAE-2 style)
- [x] Jukebox-style level training (Bottom / Middle / Top)
- [x] Conditional PixelCNN prior
- [x] Hierarchical Conditional PixelCNN prior
- [ ] Larger multi-instrument dataset
- [ ] Genre-specific training sets

## Core Idea (Short Version)

1. **Encoder + Quantizer (VQ-VAE)** compresses spectrograms into discrete code indices.
2. **Prior (PixelCNN)** learns to autoregressively sample new index grids.
3. **Decoder (VQ-VAE)** reconstructs spectrograms from sampled indices.
4. **Vocoder / Griffin-Lim** converts generated spectrograms back to waveform.

This split lets you model structure in latent space while keeping generation stable and high quality.