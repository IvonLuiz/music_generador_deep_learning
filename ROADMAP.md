# Project Roadmap

Updated roadmap with what has already been implemented and the next steps to better match the Jukebox paper.

## ✅ Already Implemented

### VQ-VAE / Autoencoders
- [x] VAE (baseline)
- [x] VQ-VAE (PyTorch)
- [x] VQ-VAE residual
- [x] Hierarchical VQ-VAE (VQ-VAE2, top+bottom)
- [x] Jukebox-style VQ-VAE trained as separate levels (bottom, middle, top)

### Priors
- [x] Conditional PixelCNN (single level)
- [x] 2-level hierarchical PixelCNN (top -> bottom)
- [x] 3-level hierarchical PixelCNN for Jukebox (top -> middle -> bottom)
- [x] Transformer top prior MVP (causal autoregressive, next-token training)
- [x] Standardized `priors` config in hierarchical workflows
- [x] Shared loading/parsing utilities for hierarchical test scripts

### Infra / code quality
- [x] Python 3.8 compatibility for type hints in new scripts
- [x] Old/new checkpoint compatibility (`model.` / `module.` prefixes)
- [x] Backward-compatible config support (legacy `top_prior`/`bottom_prior` + `priors.*`)

## 🚧 In Progress / Next Focus

### 1) Improve audio quality (high priority)
- [ ] Review generation denormalization range (`min_db`, `max_db`) per dataset/file
- [x] Add temperature and top-k for prior sampling
- [x] Add stage-wise debug mode:
- [x] real top/middle + generated bottom
- [x] real top + generated middle/bottom
- [x] generated top/middle/bottom

Reduce top compression (fewer downsamples; bigger top grid).
Add EMA codebook updates + dead-code restart (improves code usage stability).
Add multi-resolution spectral loss (critical for audio realism).
Run staged diagnostics:
real top+middle → generate bottom
real top → generate middle+bottom
fully generated top+middle+bottom
This isolates where noise explodes.

### 2) Closer alignment with the Jukebox paper (high priority)
- [x] Implement **Transformer top prior (MVP)**
- [ ] Implement **Transformer middle upsampler** conditioned on top
- [ ] Implement **Transformer bottom upsampler** conditioned on middle
- [ ] Implement cascading sampling with sliding windows (windowed sampling)
- [ ] Implement primed sampling (continuation from real audio)

### 3) Improve VQ-VAE to reduce structural noise (high priority)
- [ ] Implement VQ with **EMA codebook updates**
- [ ] Implement **dead-code restart** (revive underused embeddings)
- [ ] Add **multi-resolution spectral loss** to VQ-VAE training
- [ ] Monitor codebook perplexity/usage by level (top/middle/bottom)

### 4) Conditioning and control (paper-like)
- [ ] Metadata conditioning (genre/artist)
- [ ] Timing conditioning (segment position / song progress)
- [ ] (Optional) lyrics conditioning (encoder-decoder)

### 5) Data and generalization
- [ ] Expand dataset beyond piano (MAESTRO)
- [ ] Create genre subsets for conditioned prior training
- [ ] Evaluate FSDD/GTZAN/new datasets with a unified pipeline

## 🎯 Suggested Milestones (short-term)

### Milestone A — “Clean audio with current pipeline”
- [ ] Temperature/top-k + stage-wise debug
- [ ] Spectral loss + EMA VQ
- [ ] Re-train bottom/middle/top VQ-VAE
- [ ] Re-train 3-level prior

### Milestone B — “Paper-like priors”
- [x] Top prior Transformer (MVP)
- [ ] Upsamplers Transformer (middle/bottom)
- [ ] Windowed + primed sampling

## ✅ Verification Checklist (must run each retrain)

- [ ] Confirm inferred latent grids in training logs (top/middle/bottom).
- [ ] Confirm top grid is not over-compressed (prefer >= 8x8 for 256x256 input).
- [ ] Confirm Jukebox models load with zero missing/unexpected keys.
- [ ] Confirm prior checkpoints are from the same VQ-VAE level_profiles generation.
- [ ] Confirm staged tests improve quality in this order:
- [ ] real top+middle -> generated bottom (best)
- [ ] real top -> generated middle+bottom (middle)
- [ ] fully generated (hardest)

## 🧪 What Needs Testing Now

- [ ] Train `train_transformer_prior_top.py` and verify val loss decreases.
- [ ] Sample top codes with `test_transformer_prior_top.py` and inspect token diversity.
- [ ] Compare fully generated audio with:
- [ ] PixelCNN top prior
- [ ] Transformer top prior + existing upsamplers
- [ ] A/B test top compression settings (3/4/5 vs 3/4/6).

## 📝 Remember

- If `level_profiles` changes, all priors must be retrained.
- Make top compression decision first, then tune `dilation_growth_rate`, then explore `channel_growth`.
- For this 2D spectrogram adaptation, perfect paper parity is not expected; track quality regressions via staged generation tests.