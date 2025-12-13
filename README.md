# Backing-track AI Generator

An on-going project to generate simple backing tracks using a VQ-VAE (Vector Quantized Variational AutoEncoder) architecture.

## Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Project Roadmap

- **VQ-VAE TODO**:
  - [ ] **Train the VQ-VAE on larger dataset**
  - [] **Fix Noisy VQ-VAE output**:
    - [x] Try residual network training
      - This improved
    - [x] Try inverse filters (32, 64, 128, 256)
      - This improved
    - [x] Increase filters size (32, 64, 128, 256) -> (62, 128, 256, 512)
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
  - **Task**: Autoregressive prediction. Given code $z_1, z_2, ... z_{t-1}$, predict code $z_t$.
  - **Loss**: Cross-Entropy Loss. Guess the correct integer code (0-512) for the next position.
- [ ] **Expand Dataset**: Currently training on piano data (MAESTRO), aim to gather data from other instruments.
- [ ] **Genre Specific**: Find blues backing tracks to train on.

## How Generation Works

1.  **Sampling - Using the Prior (The "Composer")**:
    *   Start with an empty grid (or a "start-of-sequence" token).
    *   Ask the Prior to predict the probability distribution for the first code.
    *   Sample from that distribution.
    *   Repeat for all time steps and frequency bins.
    *   *Result*: A brand new grid of indices $z_{generated}$ that never existed in the dataset.

2.  **Decoding - Using the Trained VQ-VAE Decoder (The "Performer")**:
    *   Take the generated indices $z_{generated}$.
    *   Look up their vectors in the VQ-VAE codebook ($e$) to get quantized vectors $z_q$.
    *   Feed $z_q$ into the Decoder.
    *   *Result*: A brand new Spectrogram. Convert it to audio using Griffin-Lim or a Vocoder.

## VQ-VAE Theory

### Loss Function

The total loss $L$ is composed of three terms:

$$ L = \underbrace{\log p(x|z_q(x))}_{\text{Reconstruction Loss}} + \underbrace{||sg[z_e(x)] - e||_2^2}_{\text{Codebook Loss}} + \underbrace{\beta ||z_e(x) - sg[e]||_2^2}_{\text{Commitment Loss}} $$

*   **Reconstruction Loss**: Makes the output sound like the input.
*   **Codebook Loss**: Moves the codebook vectors ($e$) closer to the encoder outputs ($z_e$).
*   **Commitment Loss**: Prevents the encoder outputs ($z_e$) from fluctuating too wildly, forcing them to commit to a codebook vector.

### Codebook Collapse and Recovery

During training, you might observe the loss decreasing, then spiking, and then decreasing again. This is a known phenomenon:

1.  **The "Easy Way Out" (Initial Drop)**:
    *   The model finds a few "good enough" embeddings (e.g., 5-10 out of 1024).
    *   The Encoder maps everything to these few codes.
    *   Reconstruction Loss drops quickly because it predicts a rough average.

2.  **The "Realization" (The Spike)**:
    *   The model realizes limited codes aren't enough for fine details (noise/texture).
    *   It forces the Encoder to use new, unused embeddings.
    *   **Conflict**: The Encoder outputs a vector for a new code that is currently random garbage. The Decoder produces a bad reconstruction.
    *   *Result*: Reconstruction and Commitment losses spike.

3.  **The "Learning" (The Second Drop)**:
    *   Gradients flow back.
    *   The Codebook updates the "garbage" embedding to match the music feature.
    *   The Encoder learns to map to it precisely.
    *   *Result*: Loss decreases again, this time with higher fidelity.

*Note: The VQ Loss (Commitment + Codebook) often "bounces" while the Reconstruction Loss generally trends down (with bumps during codebook shifts).*