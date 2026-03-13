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