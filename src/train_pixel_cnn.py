import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime
import yaml

from modeling.torch.pixel_cnn import ConditionalGatedPixelCNN
from modeling.torch.vq_vae import VQ_VAE
from modeling.torch.vq_vae_residual import VQ_VAE as VQ_VAE_Residual
from datasets.quantized_dataset import QuantizedDataset
from utils import load_maestro, load_config
from processing.preprocess_audio import TARGET_TIME_FRAMES

def train_pixel_cnn(pixelcnn_config_path: str, vqvae_model_path: str):
    # Load PixelCNN configuration
    pixelcnn_config = load_config(pixelcnn_config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    # Load VQ-VAE configuration (assumed to be in the same directory as the model)
    vqvae_config_path = os.path.join(os.path.dirname(vqvae_model_path), "config.yaml")
    if not os.path.exists(vqvae_config_path):
        # Fallback to default VQ-VAE config if run-specific config not found
        print(f"Warning: Run-specific config not found at {vqvae_config_path}. Using default config/config_vqvae.yaml")
        vqvae_config_path = "./config/config_vqvae.yaml"
    
    vqvae_config = load_config(vqvae_config_path)
    print(f"Loaded VQ-VAE config from {vqvae_config_path}")

    # VQ-VAE Parameters
    K = vqvae_config['model']['K']
    D = vqvae_config['model']['D']
    conv_filters = tuple(vqvae_config['model']['conv_filters'])
    conv_kernels = tuple(vqvae_config['model']['conv_kernels'])
    conv_strides = tuple([tuple(s) for s in vqvae_config['model']['conv_strides']])
    dropout_rate_vqvae = vqvae_config['model'].get('dropout_rate', 0.0)
    use_residual = vqvae_config['model'].get('use_residual', False)

    # PixelCNN Parameters
    BATCH_SIZE = pixelcnn_config['training']['batch_size']
    LEARNING_RATE = pixelcnn_config['training']['learning_rate']
    EPOCHS = pixelcnn_config['training']['epochs']
    SPECTROGRAMS_PATH = pixelcnn_config['dataset']['processed_path']
    SAVE_DIR = pixelcnn_config['training']['save_dir']
    MODEL_NAME = pixelcnn_config['model']['name']
    
    HIDDEN_CHANNELS = pixelcnn_config['model']['hidden_channels']
    NUM_LAYERS = pixelcnn_config['model']['num_layers']
    KERNEL_SIZE = pixelcnn_config['model']['kernel_size']
    DROPOUT_RATE_PIXELCNN = pixelcnn_config['model'].get('dropout_rate', 0.0)

    # Load VQ-VAE Model
    print(f"Loading VQ-VAE model from {vqvae_model_path}")
    if use_residual:
        vqvae = VQ_VAE_Residual(
            input_shape=(256, TARGET_TIME_FRAMES, 1),
            conv_filters=conv_filters,
            conv_kernels=conv_kernels,
            conv_strides=conv_strides,
            latent_space_dim=D,
            embeddings_size=K,
            dropout_rate=dropout_rate_vqvae
        )
    else:
        vqvae = VQ_VAE(
            input_shape=(256, TARGET_TIME_FRAMES, 1),
            conv_filters=conv_filters,
            conv_kernels=conv_kernels,
            conv_strides=conv_strides,
            latent_space_dim=D,
            embeddings_size=K,
            dropout_rate=dropout_rate_vqvae
        )
    checkpoint = torch.load(vqvae_model_path, map_location=device)
    vqvae.load_state_dict(checkpoint['model_state'])
    vqvae.to(device)
    vqvae.eval()

    # Create Dataset (Extract Codes)
    x_train, _ = load_maestro(SPECTROGRAMS_PATH, TARGET_TIME_FRAMES)
    dataset = QuantizedDataset(x_train, vqvae, device)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize PixelCNN
    # We model the indices (0 to K-1).
    # Input channels = 1 (indices).
    # Num classes = K.
    # We use embeddings, so we pass num_embeddings=K.
    pixel_cnn = ConditionalGatedPixelCNN(
        in_channels=1,
        hidden_channels=HIDDEN_CHANNELS,
        num_layers=NUM_LAYERS,
        kernel_size=KERNEL_SIZE,
        num_classes=K,
        num_embeddings=K,     # Enable embedding layer
        dropout_rate=DROPOUT_RATE_PIXELCNN
    ).to(device)

    optimizer = optim.Adam(pixel_cnn.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path_dir = os.path.join(SAVE_DIR, f"{MODEL_NAME}", current_time)
    os.makedirs(save_path_dir, exist_ok=True)
    
    # Save the pixelcnn config for reproducibility
    with open(os.path.join(save_path_dir, "config.yaml"), 'w') as f:
        yaml.dump(pixelcnn_config, f)
    
    print(f"Starting PixelCNN training for {EPOCHS} epochs...")
    
    for epoch in range(1, EPOCHS + 1):
        pixel_cnn.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
        
        for batch_indices in pbar:
            batch_indices = batch_indices.to(device) # (B, H, W)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = pixel_cnn(batch_indices) # (B, K, 1, H, W)
            output = output.squeeze(2) # (B, K, H, W)
            
            # Loss
            # CrossEntropyLoss expects (B, K, H, W) and Target (B, H, W)
            loss = criterion(output, batch_indices)
            
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(pixel_cnn.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if epoch % 10 == 0 or epoch == EPOCHS:
            save_file = os.path.join(save_path_dir, f"pixelcnn_epoch_{epoch}.pth")
            torch.save({
                'model_state': pixel_cnn.state_dict(),
                'config': pixelcnn_config,
                'epoch': epoch,
                'loss': avg_loss
            }, save_file)
            print(f"Saved checkpoint to {save_file}")

    print("Training complete.")

if __name__ == "__main__":
    # Default paths
    PIXELCNN_CONFIG_PATH = "./config/config_pixelcnn.yaml"
    VQVAE_CONFIG_PATH = "./config/config_vqvae.yaml" # Used to find save_dir
    
    # Find latest VQ-VAE model
    vqvae_global_config = load_config(VQVAE_CONFIG_PATH)
    LATEST_MODEL_POINTER = os.path.join(vqvae_global_config['training']['save_dir'], "latest_model_path.txt")
    
    if os.path.exists(LATEST_MODEL_POINTER):
        with open(LATEST_MODEL_POINTER, 'r') as f:
            MODEL_PATH = f.read().strip()
        print(f"Using latest VQ-VAE model: {MODEL_PATH}")
        train_pixel_cnn(PIXELCNN_CONFIG_PATH, MODEL_PATH)
    else:
        print("No latest model found. Please provide path manually or train VQ-VAE first.")
