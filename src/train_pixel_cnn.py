import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime
import yaml
import matplotlib.pyplot as plt

from modeling.torch.pixel_cnn import ConditionalGatedPixelCNN
from datasets.quantized_dataset import QuantizedDataset
from utils import load_maestro, load_config, load_vqvae_model
from processing.preprocess_audio import TARGET_TIME_FRAMES

def plot_pixelcnn_losses(train_losses, val_losses, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PixelCNN Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(save_dir, 'loss_plot.png')
    plt.savefig(save_path)
    plt.close()

def train_pixel_cnn(pixelcnn_config_path: str, vqvae_model_path: str):
    # Load PixelCNN configuration
    pixelcnn_config = load_config(pixelcnn_config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    # Load VQ-VAE configuration (assumed to be in the same directory as the model)
    # But we still need K for PixelCNN initialization
    print(f"Loading VQ-VAE model from {vqvae_model_path}")
    vqvae = load_vqvae_model(vqvae_model_path, device)
    K = vqvae.embeddings.num_embeddings
    print(f"Extracted K={K} from VQ-VAE model")

    # PixelCNN Parameters
    batch_size = pixelcnn_config['training']['batch_size']
    learning_rate = pixelcnn_config['training']['learning_rate']
    epochs = pixelcnn_config['training']['epochs']
    validation_split = pixelcnn_config['training'].get('validation_split', 0.1)
    spectrograms_path = pixelcnn_config['dataset']['processed_path']
    save_dir = pixelcnn_config['training']['save_dir']
    model_name = pixelcnn_config['model']['name']
    
    hidden_channels = pixelcnn_config['model']['hidden_channels']
    num_layers = pixelcnn_config['model']['num_layers']
    kernel_size = pixelcnn_config['model']['kernel_size']

    # Creating Datasets
    # load data
    spectrograms_data, _ = load_maestro(spectrograms_path, TARGET_TIME_FRAMES)
    # split into train/val
    num_samples = len(spectrograms_data)
    num_val = int(num_samples * validation_split)
    num_train = num_samples - num_val
    # shuffle indices
    indices = np.random.permutation(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    x_train = spectrograms_data[train_indices]
    x_val = spectrograms_data[val_indices]
    print(f"Data split: {len(x_train)} training, {len(x_val)} validation samples.")
    
    print("Creating Training Dataset...")
    train_dataset = QuantizedDataset(x_train, vqvae, device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print("Creating Validation Dataset...")
    val_dataset = QuantizedDataset(x_val, vqvae, device)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize PixelCNN
    # We model the indices (0 to K-1).
    # Input channels = 1 (indices).
    # Num classes = K.
    # We use embeddings, so we pass num_embeddings=K.
    pixel_cnn = ConditionalGatedPixelCNN(
        in_channels=1,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        kernel_size=kernel_size,
        num_classes=K,
        num_embeddings=K,     # Enable embedding layer
    ).to(device)

    optimizer = optim.Adam(pixel_cnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path_dir = os.path.join(save_dir, f"{model_name}", current_time)
    os.makedirs(save_path_dir, exist_ok=True)

    # Inject K into pixelcnn_config before saving, so load_pixelcnn_model can find it
    pixelcnn_config['model']['K'] = K

    # Save the pixelcnn config for reproducibility
    with open(os.path.join(save_path_dir, "config.yaml"), 'w') as f:
        yaml.dump(pixelcnn_config, f)
    
    print(f"Starting PixelCNN training for {epochs} epochs...")
    
    best_model_path = os.path.join(save_path_dir, "best_pixelcnn_model.pth")
    best_val_loss = float('inf')
    
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        # Training
        pixel_cnn.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        
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
            
            total_train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        pixel_cnn.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_indices in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                batch_indices = batch_indices.to(device)
                output = pixel_cnn(batch_indices)
                output = output.squeeze(2)
                loss = criterion(output, batch_indices)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Logging and plotting
        print(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        plot_pixelcnn_losses(train_losses, val_losses, save_path_dir)
        
        # Save checkpoint
        if epoch % 10 == 0 or epoch == epochs:
            save_file = os.path.join(save_path_dir, f"pixelcnn_epoch_{epoch}.pth")
            torch.save({
                'model_state': pixel_cnn.state_dict(),
                'config': pixelcnn_config,
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, save_file)
            print(f"Saved checkpoint to {save_file}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state': pixel_cnn.state_dict(),
                'config': pixelcnn_config,
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, best_model_path)
            print(f"Saved best model on validation data to {best_model_path} on epoch {epoch}")

    print("Training complete.")

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

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
