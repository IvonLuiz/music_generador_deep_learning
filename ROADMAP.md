# Legacy Roadmap Notes

Archived planning and TODO notes from the previous README.

## Project Roadmap (Legacy)

- **VQ-VAE TODO**:
  - [X] VAE
  - [X] VQVAE
  - [X] VQVAE residual
  - [X] VQVAE Hierarchical (VQVAE2)
  - [ ] **Train the VQ-VAE on larger dataset**
  - [x] **Fix Noisy VQ-VAE output**:
    - [x] Try residual network training
      - This improved
    - [x] Try inverse filters (32, 64, 128, 256)
      - This improved
    - [x] Increase filters size (32, 64, 128, 256) -> (62, 128, 256, 512)
  - [x] VQ_VAE2:
    - [x] This really improves the noisy
  - [ ] Increase latent space:
    - [ ] Decrease strides
- **Prior (The "Composer") model TODO**
  - Implement Prior archtectures:
    - [ ] PixelCNN
    - [X] CondicionalPixelCNN
    - [ ] WaveNet
    - [ ] Transformer GPT like
  - Train Prior:
    - [ ] PixelCNN
    - [X] CondicionalPixelCNN
      - [ ] Need a dataset with vectors for different genres
    - [ ] WaveNet
    - [ ] Transformer GPT like
  - The paper suggests using a PixelCNN for images (2D data like spectrograms) or a WaveNet for raw audio.
  - Potential approach: Modern Transformer (like GPT).
  - **Task**: Autoregressive prediction. Given code z_1, z_2, ... z_{t-1}, predict code z_t.
  - **Loss**: Cross-Entropy Loss. Guess the correct integer code (0-512) for the next position.
- [ ] **Expand Dataset**: Currently training on piano data (MAESTRO), aim to gather data from other instruments.
- [ ] **Genre Specific**: Find blues backing tracks to train on.
