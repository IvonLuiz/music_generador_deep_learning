# Backing-track AI Generator

An on-going project to generate simple backing tracks using a VQ-VAE (Vector Quantized Variational AutoEncoder) architecture.

## Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train Jukebox VQ-VAE (Bottom → Middle → Top)

Run all 3 Jukebox VQ-VAE levels sequentially:

```bash
bash src/train_scripts/train_vqvae_jukebox_all_levels.sh
```

Optional: pass extra arguments through to each run (for example, any future trainer flags):

```bash
bash src/train_scripts/train_vqvae_jukebox_all_levels.sh <extra-args>
```

## Project Roadmap

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

## Theoretical Background

### 1. VAE (Variational AutoEncoder)
A standard VAE learns a continuous latent space where inputs are mapped to probability distributions (usually Gaussians) rather than fixed points.
- **Encoder**: Predicts the mean ($\mu$) and variance ($\sigma$) of the distribution for a given input.
- **Sampling**: A latent vector $z$ is sampled from this distribution: $z = \mu + \sigma \cdot \epsilon$ (where $\epsilon$ is random noise).
- **Decoder**: Reconstructs the input from the sampled $z$.
- **KL Divergence**: A loss term that forces the learned distributions to be close to a standard Normal distribution $\mathcal{N}(0, 1)$.

While powerful for generation, VAEs often produce "blurry" results because the model averages over the noise in the latent space.

### 3. VQ-VAE (Vector Quantized Variational AutoEncoder)
![VQ-VAE Architecture](images/vq_vae.png)

The VQ-VAE combines standard AutoEncoders with Vector Quantization to learn a discrete latent representation of the data.
Vector Quantization is a technique to map continuous vectors from a high-dimensional space into a finite set of discrete vectors called a **Codebook**.

This process is non-differentiable (due to the `argmin` operation), so we use the "Straight-Through Estimator" trick during training, allowing gradients to bypass the quantization step.

- **Encoder**: Compresses the input (e.g., a spectrogram) into a smaller grid of vectors.
- **Quantization**: Each vector in the grid is replaced by the nearest neighbor from the Codebook. This results in a grid of indices (discrete codes) or learnable vectors $\{e_1, e_2, ..., e_K\}$.
- **Decoder**: Takes the quantized vectors and reconstructs the original input.

This discrete representation allows us to use powerful autoregressive models (like PixelCNN or Transformers) to generate new music by predicting sequences of these codes.

#### Loss Function

The total loss $L$ is composed of three terms:

$$ L = \underbrace{\log p(x|z_q(x))}_{\text{Reconstruction Loss}} + \underbrace{||sg[z_e(x)] - e||_2^2}_{\text{Codebook Loss}} + \underbrace{\beta ||z_e(x) - sg[e]||_2^2}_{\text{Commitment Loss}} $$

*   **Reconstruction Loss**: Makes the output sound like the input.
*   **Codebook Loss**: Moves the codebook vectors ($e$) closer to the encoder outputs ($z_e$).
*   **Commitment Loss**: Prevents the encoder outputs ($z_e$) from fluctuating too wildly, forcing them to commit to a codebook vector.

*Note: The VQ Loss (Commitment + Codebook) often "bounces" while the Reconstruction Loss generally trends down (with bumps during codebook shifts).*

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

### 4. VQ-VAE-2 (Hierarchical VQ-VAE)
![VQ-VAE-2 Architecture](images/vq_vae2.jpg)

The VQ-VAE-2 improves upon the original by using a **hierarchical structure** with multiple layers of latent variables.
- **Bottom Level**: Captures local details (texture, timbre). It has a higher resolution.
- **Top Level**: Captures global structure (melody, rhythm, form). It is more compressed (lower resolution).
- **Training**: The encoder passes information up to the Top level. The Top level is quantized and then passed back down to condition the Bottom level.

This separation allows the model to generate high-fidelity audio with coherent long-term structure, addressing the "noisiness" or lack of structure often seen in single-layer models.

#### Loss

The total loss is the sum of the reconstruction loss and the VQ losses (codebook + commitment) for each level of the hierarchy.

$$ L = \underbrace{\log p(x|z_{q,top}, z_{q,bottom})}_{\text{Reconstruction Loss}} + \sum_{level} \left( \underbrace{||sg[z_{e,level}] - e_{level}||_2^2}_{\text{Codebook Loss}} + \underbrace{\beta ||z_{e,level} - sg[e_{level}]||_2^2}_{\text{Commitment Loss}} \right) $$
