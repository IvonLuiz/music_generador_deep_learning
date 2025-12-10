import os
import argparse
import torch
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

from generation.soundgenerator import SoundGenerator
from generation.generate import save_multiple_signals
from utils import load_config, load_vqvae_model, load_pixelcnn_model
from processing.preprocess_audio import TARGET_TIME_FRAMES

def generate_pixelcnn_samples(pixelcnn_model, num_samples, latent_shape, device):
    """
    Generate samples using PixelCNN autoregressively.
    """
    pixelcnn_model.eval()
    
    # Initialize blank canvas with zeros (or random starting point if needed)
    # Shape: (B, H, W) for indices
    H, W = latent_shape
    samples = torch.zeros((num_samples, H, W), dtype=torch.long).to(device) # (B, H, W)
    
    print(f"Generating {num_samples} samples of shape {latent_shape}...")
    
    with torch.no_grad():
        for i in tqdm(range(H), desc="Generating rows"):
            for j in range(W):
                # Forward pass
                logits = pixelcnn_model(samples) # (B, K, 1, H, W)
                if logits.ndim == 5: # (B, K, C, H, W)
                    logits = logits.squeeze(2) # (B, K, H, W)
                
                # Get logits for the current pixel position (i, j)
                pixel_logits = logits[:, :, i, j] # (B, K)
                
                # Sample from the distribution by using multinomial
                probs = F.softmax(pixel_logits, dim=-1)
                next_pixel = torch.multinomial(probs, 1).squeeze(1) # (B,)
                
                # Update samples
                samples[:, i, j] = next_pixel
                
    return samples

def decode_indices(indices, vqvae_model):
    """
    Decode indices (B, H, W) to spectrograms using the trained VQ-VAE decoder.
    """
    vqvae_model.eval()
    
    with torch.no_grad():
        # Convert indices to embeddings
        # vqvae.vq.embedding is (K, D)
        # indices is (B, H, W)
        
        # F.embedding(indices, weight) -> (B, H, W, D)
        z_q = F.embedding(indices, vqvae_model.vq.embedding) # (B, H, W, D)
        z_q = z_q.permute(0, 3, 1, 2).contiguous() # (B, D, H, W)
        
        # Decode
        x_hat = vqvae_model.decoder(z_q) # (B, 1, H, W)
        spectrograms = x_hat.permute(0, 2, 3, 1).cpu().numpy() # (B, H, W, 1)
        
    return spectrograms

def test_pixel_cnn(pixelcnn_model_path: str, vqvae_model_path: str, num_samples: int = 5,
                   default_min_db: float = -40.0, default_max_db: float = 40.0):
    """
    Test PixelCNN model by generating samples and converting them to audio.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    print(f"Loading VQ-VAE from {vqvae_model_path}")
    vqvae = load_vqvae_model(vqvae_model_path, device)
    
    # Get K from VQ-VAE to help PixelCNN loading if config is missing it
    # VQ-VAE has a VectorQuantizer at self.vq
    K = vqvae.vq.num_embeddings
    print(f"Extracted K={K} from VQ-VAE model")
    
    print(f"Loading PixelCNN from {pixelcnn_model_path}")
    pixel_cnn = load_pixelcnn_model(pixelcnn_model_path, device, num_embeddings=K)
    
    # Determine latent shape from VQ-VAE config or by running a dummy input through the encoder
    dummy_input = torch.zeros((1, 1, 256, TARGET_TIME_FRAMES)).to(device)
    with torch.no_grad():
        z_e = vqvae.encoder(dummy_input)
        latent_shape = z_e.shape[2:] # (H, W)
    print(f"Latent shape determined: {latent_shape}")

    # Generate samples
    generated_indices = generate_pixelcnn_samples(pixel_cnn, num_samples, latent_shape, device)
    
    print("Decoding indices to spectrograms...")
    generated_spectrograms = decode_indices(generated_indices, vqvae)
    
    # Convert to audio
    print("Converting spectrograms to audio...")
    # We need hop_length from config
    # Try to load config from vqvae path
    if os.path.isdir(vqvae_model_path):
        config_path = os.path.join(vqvae_model_path, "config.yaml")
    else:
        config_path = os.path.join(os.path.dirname(vqvae_model_path), "config.yaml")
    
    config = load_config(config_path)
    hop_length = config['dataset']['hop_length']
    
    sound_generator = SoundGenerator(vqvae, hop_length=hop_length)
    
    # The model outputs [0, 1], so we need to map back to dB denormalized.
    min_max_values = [{"min": default_min_db, "max": default_max_db} for _ in range(num_samples)]
    signals = sound_generator.convert_spectrograms_to_audio(generated_spectrograms, min_max_values)
    
    # Save results and spectrogram plots
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"samples/pixelcnn_generated/{current_time}/"
    
    print(f"Saving samples to {save_dir}")
    save_multiple_signals({'generated': signals}, save_dir)
    
    os.makedirs(os.path.join(save_dir, "spectrograms"), exist_ok=True)
    for i, spec in enumerate(generated_spectrograms):
        plt.figure(figsize=(10, 4))
        plt.imshow(spec[:, :, 0], origin='lower', aspect='auto')
        plt.colorbar()
        plt.title(f"Generated Spectrogram {i}")
        plt.savefig(os.path.join(save_dir, "spectrograms", f"spec_{i}.png"))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PixelCNN model")
    parser.add_argument("--vqvae_path", type=str, default="models/vq_vae/2025-11-24_00-17-34", help="Path to VQ-VAE model")
    parser.add_argument("--pixelcnn_path", type=str, default="models/pixelcnn_maestro2011/2025-11-30_17-57-20", help="Path to PixelCNN model")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--min_db", type=float, default=-40.0, help="Minimum dB value")
    parser.add_argument("--max_db", type=float, default=40.0, help="Maximum dB value")
    
    args = parser.parse_args()
    
    VQVAE_PATH = args.vqvae_path
    PIXELCNN_PATH = args.pixelcnn_path
    
    # Check if paths exist, if not, try to find them or warn user
    if not os.path.exists(VQVAE_PATH):
        print(f"Warning: VQ-VAE path {VQVAE_PATH} does not exist. Please provide a valid path.")
        exit(1)
    if not os.path.exists(PIXELCNN_PATH):
        print(f"Warning: PixelCNN path {PIXELCNN_PATH} does not exist. Please provide a valid path.")
        exit(1)
        
    test_pixel_cnn(PIXELCNN_PATH, VQVAE_PATH, num_samples=args.num_samples, default_min_db=args.min_db, default_max_db=args.max_db)