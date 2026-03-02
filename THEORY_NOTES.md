# Legacy Theory Notes

Archived long-form background notes from the previous README.

## How Generation Works (Legacy)

1. **Sampling - Using the Prior (The "Composer")**:
   - Start with an empty grid (or a "start-of-sequence" token).
   - Ask the Prior to predict the probability distribution for the first code.
   - Sample from that distribution.
   - Repeat for all time steps and frequency bins.
   - Result: A brand new grid of indices z_generated that never existed in the dataset.

2. **Decoding - Using the Trained VQ-VAE Decoder (The "Performer")**:
   - Take the generated indices z_generated.
   - Look up their vectors in the VQ-VAE codebook (e) to get quantized vectors z_q.
   - Feed z_q into the Decoder.
   - Result: A brand new Spectrogram. Convert it to audio using Griffin-Lim or a Vocoder.

## Theoretical Background (Legacy)

### 1. VAE (Variational AutoEncoder)

A standard VAE learns a continuous latent space where inputs are mapped to probability distributions (usually Gaussians) rather than fixed points.

- **Encoder**: Predicts the mean (mu) and variance (sigma) of the distribution for a given input.
- **Sampling**: A latent vector z is sampled from this distribution: z = mu + sigma * epsilon (where epsilon is random noise).
- **Decoder**: Reconstructs the input from the sampled z.
- **KL Divergence**: A loss term that forces the learned distributions to be close to a standard Normal distribution N(0, 1).

While powerful for generation, VAEs often produce "blurry" results because the model averages over the noise in the latent space.

### 3. VQ-VAE (Vector Quantized Variational AutoEncoder)

The VQ-VAE combines standard AutoEncoders with Vector Quantization to learn a discrete latent representation of the data.
Vector Quantization maps continuous vectors from a high-dimensional space into a finite set of discrete vectors called a **Codebook**.

This process is non-differentiable (due to argmin), so the Straight-Through Estimator trick is used during training, allowing gradients to bypass quantization.

- **Encoder**: Compresses input (e.g., spectrogram) into a smaller grid of vectors.
- **Quantization**: Each vector is replaced by its nearest codebook vector.
- **Decoder**: Reconstructs the input from quantized vectors.

This discrete representation enables autoregressive models (PixelCNN/Transformers) to generate new music by predicting sequences of codes.

#### Loss Function

Total loss L is composed of three terms:

L = Reconstruction Loss + Codebook Loss + Commitment Loss

- **Reconstruction Loss**: Makes output sound like input.
- **Codebook Loss**: Moves codebook vectors toward encoder outputs.
- **Commitment Loss**: Prevents encoder outputs from fluctuating too much.

Note: VQ loss (commitment + codebook) can bounce while reconstruction generally trends down with bumps.

### Codebook Collapse and Recovery

During training, loss can decrease, then spike, then decrease again:

1. **The Easy Way Out (Initial Drop)**
   - Model uses only a few embeddings.
   - Reconstruction drops quickly via rough averages.

2. **The Realization (Spike)**
   - Limited codes are insufficient for detail.
   - New embeddings are used but start random, causing temporary bad reconstructions.

3. **The Learning (Second Drop)**
   - Gradients align encoder outputs and codebook vectors.
   - Loss decreases again with better fidelity.

### 4. VQ-VAE-2 (Hierarchical VQ-VAE)

VQ-VAE-2 improves VQ-VAE using a hierarchy of latent variables:

- **Bottom Level**: Captures local details (texture/timbre), higher resolution.
- **Top Level**: Captures global structure (melody/rhythm/form), lower resolution.
- **Training**: Information goes up to top level, top is quantized, then conditions lower-level decoding.

This helps reduce noisy outputs and improves long-range coherence.

#### Hierarchical Loss

Total loss is reconstruction + VQ losses for each level:

L = Reconstruction(x | z_q_top, z_q_bottom) + sum over levels (codebook + commitment)
