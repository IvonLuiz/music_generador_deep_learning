import os
import argparse
import torch
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generation.soundgenerator import SoundGenerator
from generation.generate import save_multiple_signals
from utils import load_config, load_vqvae_hierarchical_model_wrapper
from modeling.torch.pixel_cnn_hierarchical import HierarchicalCondGatedPixelCNN
from processing.preprocess_audio import TARGET_TIME_FRAMES

def load_hierarchical_pixelcnn_model(model_dir: str, device: torch.device, weights_file="best_model.pth"):
    """
    Loads a trained Hierarchical PixelCNN model and its config.
    """
    if os.path.isfile(model_dir):
        config_path = os.path.join(os.path.dirname(model_dir), 'config.yaml')
        model_path = model_dir
    else:
        config_path = os.path.join(model_dir, 'config.yaml')
        model_path = os.path.join(model_dir, weights_file)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    config = load_config(config_path)
    
    # Reconstruct the model parameters as done in training
    top_cfg = config['top_prior']
    bot_cfg = config['bottom_prior']
    
    hidden_units = [top_cfg['hidden_channels'], bot_cfg['hidden_channels']]
    num_layers = [top_cfg['num_layers'], bot_cfg['num_layers']]
    conv_filter_size = [top_cfg['conv_filter_size'], bot_cfg['conv_filter_size']]
    dropout = [top_cfg.get('dropout_rate', 0.0), bot_cfg.get('dropout_rate', 0.0)]
    
    print(f"Loading Hierarchical PixelCNN weights from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # If config was saved in checkpoint (as done in recent scripts), prefer that
    if 'config' in checkpoint:
        config = checkpoint['config']
        
    state_dict = checkpoint['model_state']
    
    try:
        num_embeddings_top = state_dict['top_prior.out_conv.weight'].shape[0] 
        # Check bottom prior if exists
        if 'bottom_level.out_conv.weight' in state_dict:
            num_embeddings_bot = state_dict['bottom_level.out_conv.weight'].shape[0]
            num_embeddings = [num_embeddings_top, num_embeddings_bot]
        else:
            num_embeddings = [num_embeddings_top, num_embeddings_top] # Fallback if single level? but this is hierarchical loader
    except KeyError:
        print("Could not infer num_embeddings from state_dict keys (top_prior.out_conv.weight not found). Using default [512, 512].")
        num_embeddings = [512, 512] 
    
    pixelcnn = HierarchicalCondGatedPixelCNN(
        num_prior_levels=2,
        input_size=[(32, 32), (64, 64)], 
        hidden_units=hidden_units,
        num_layers=num_layers,
        conv_filter_size=conv_filter_size,
        dropout=dropout,
        num_embeddings=num_embeddings,
        residual_units=[1024, 1024], 
        attention_layers=[0, 0],
        attention_heads=[None, None],
        conditioning_stack_residual_blocks=[None, 20] 
    ).to(device)
    
    # Load weights
    pixelcnn.load_state_dict(state_dict)
        
    pixelcnn.eval()
    return pixelcnn

def generate_hierarchical_samples(pixelcnn_model: HierarchicalCondGatedPixelCNN, num_samples: int, latent_shapes: list[tuple[int, int]], device: torch.device):
    """
    Generate samples using Hierarchical PixelCNN.
    latent_shapes: List of tuples [(H_top, W_top), (H_bot, W_bot)]
    """
    pixelcnn_model.eval()
    
    top_shape = latent_shapes[0]
    bot_shape = latent_shapes[1]

    H_top, W_top = top_shape
    H_bot, W_bot = bot_shape

    with torch.no_grad():
        print(f"Generating Top Level ({num_samples} samples, shape {top_shape})...")
        top_samples = pixelcnn_model.generate(shape=(num_samples, 1, H_top, W_top), level='top').squeeze(1)
        print(f"Generating Bottom Level ({num_samples} samples, shape {bot_shape})...")
        bot_samples = pixelcnn_model.generate(shape=(num_samples, 1, H_bot, W_bot), cond=top_samples, level='bottom').squeeze(1)

    return top_samples, bot_samples

def decode_hierarchical(top_indices, bot_indices, vqvae_model):
    """
    Decode hierarchical indices to spectrograms using VQ-VAE-2 decoder.
    """
    vqvae_model.eval()
    with torch.no_grad():
        # Embed
        ## Top
        z_q_top = F.embedding(top_indices, vqvae_model.vq_top.embedding) # (B, H, W, D)
        z_q_top = z_q_top.permute(0, 3, 1, 2).contiguous()
        
        ## Bottom
        z_q_bot = F.embedding(bot_indices, vqvae_model.vq_bottom.embedding)
        z_q_bot = z_q_bot.permute(0, 3, 1, 2).contiguous()
        
        # Decode      
        ## Upsample Top Latents
        dec_t = vqvae_model.decoder_top(z_q_top)
        
        ## Concatenate with Bottom Latents
        ## IMPORTANT: Must match the order in VQ_VAE_Hierarchical.forward
        ## forward does: torch.cat([z_bottom_q, z_top_upsampled], dim=1)
        input_b = torch.cat([z_q_bot, dec_t], dim=1)
        
        ## Final Decode
        x_recon = vqvae_model.decoder_bottom(input_b)
        
        spectrograms = torch.sigmoid(x_recon).permute(0, 2, 3, 1).cpu().numpy()
        
    return spectrograms

def test_hierarchical_pixelcnn(pixelcnn_path, vqvae_path, num_samples=3, 
                               min_db=-80.0, max_db=0.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading VQ-VAE from {vqvae_path}")
    vqvae = load_vqvae_hierarchical_model_wrapper(vqvae_path, device)
    print(f"Loading PixelCNN from {pixelcnn_path}")
    hierarchical_pixelcnn = load_hierarchical_pixelcnn_model(pixelcnn_path, device)
        
    # Determine shapes by running dummy input
    dummy = torch.zeros((1, 1, TARGET_TIME_FRAMES, TARGET_TIME_FRAMES)).to(device)
    with torch.no_grad():
        enc_b = vqvae.encoder_bottom(dummy)
        enc_t = vqvae.encoder_top(enc_b)

        quant_t = vqvae.pre_vq_conv_top(enc_t)
        quant_b = vqvae.pre_vq_conv_bottom(enc_b)
        
        H_top, W_top = quant_t.shape[2], quant_t.shape[3]
        H_bot, W_bot = quant_b.shape[2], quant_b.shape[3]
        
    latent_shapes = [(H_top, W_top), (H_bot, W_bot)]
    print(f"Latent shapes determined: Top {latent_shapes[0]}, Bottom {latent_shapes[1]}")

    # Generate
    top_indices, bot_indices = generate_hierarchical_samples(hierarchical_pixelcnn, num_samples, latent_shapes, device)
    
    # Decode
    print("Decoding...")
    gen_specs = decode_hierarchical(top_indices, bot_indices, vqvae)
    
    # Save Audio & Plots
    if os.path.isdir(vqvae_path):
        c_path = os.path.join(vqvae_path, 'config.yaml')
    else:
        c_path = os.path.join(os.path.dirname(vqvae_path), 'config.yaml')
    
    try:
        cfg = load_config(c_path)
        hop_length = cfg['dataset'].get('hop_length', 256) # Default 256 if missing
    except:
        hop_length = 256
        print("Warning: Could not load VQ-VAE config for hop_length, using default 256")
        
    sound_generator = SoundGenerator(vqvae, hop_length=hop_length)
    min_max_values = [{"min": min_db, "max": max_db} for _ in range(num_samples)]
    
    signals = sound_generator.convert_spectrograms_to_audio(gen_specs, min_max_values)
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"samples/pixelcnn_hierarchical_generated/{current_time}/"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Saving to {save_dir}")
    save_multiple_signals({'generated': signals}, save_dir)
    
    spec_dir = os.path.join(save_dir, "spectrograms")
    os.makedirs(spec_dir, exist_ok=True)
    
    for i, spec in enumerate(gen_specs):
        plt.figure(figsize=(10, 4))
        plt.imshow(spec[:, :, 0], origin='lower', aspect='auto')
        plt.colorbar()
        plt.title(f"Gen Hierarchical {i}")
        plt.savefig(os.path.join(spec_dir, f"sample_{i}.png"))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pixelcnn", type=str, required=True, help="Path to Hierarchical PixelCNN model directory or .pth")
    parser.add_argument("--vqvae", type=str, required=True, help="Path to Hierarchical VQ-VAE model directory or .pth")
    parser.add_argument("--n_samples", type=int, default=3)
    parser.add_argument("--min_db", type=float, default=-80.0)
    parser.add_argument("--max_db", type=float, default=0.0)
    args = parser.parse_args()
    
    test_hierarchical_pixelcnn(args.pixelcnn, args.vqvae, args.n_samples, args.min_db, args.max_db)
