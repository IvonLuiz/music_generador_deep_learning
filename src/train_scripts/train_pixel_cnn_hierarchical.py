import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from datetime import datetime
import yaml
import matplotlib.pyplot as plt
import sys

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling.torch.pixel_cnn_hierarchical import HierarchicalCondGatedPixelCNN
from datasets.hierarchical_quantized_dataset import HierarchicalQuantizedDataset
from utils import load_maestro, load_config, initialize_vqvae_hierarchical_model, load_vqvae_hierarchical_model_wrapper
from processing.preprocess_audio import TARGET_TIME_FRAMES


def plot_losses(train_losses, val_losses, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Hierarchical PixelCNN Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()

def train_pixel_cnn_hierarchical(pixelcnn_config_path: str, vqvae_model_path: str = None):
    # Load PixelCNN configuration
    config = load_config(pixelcnn_config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    spectrograms_path = config['dataset']['processed_path']
    target_time_frames = config['dataset']['target_time_frames']
    val_split = config['training'].get('validation_split', 0.1)
    batch_size = config['training']['batch_size']
    top_cfg = config['top_prior']
    bot_cfg = config['bottom_prior']

    # Load VQ-VAE
    if vqvae_model_path is None:
        # Construct path from config
        if 'vqvae' in config and 'model_dir' in config['vqvae'] and 'specific_run_dir' in config['vqvae']:
            vqvae_model_path = os.path.join(
                config['vqvae']['model_dir'], 
                config['vqvae']['specific_run_dir']
            )
            weights_file = config['vqvae'].get('weights_file', 'best_model.pth')
            # Check if explicit weights file is in the path or just a directory
            if not os.path.isfile(vqvae_model_path):
                 # It's a directory, let load_vqvae_hierarchical_model_wrapper handle it or append weights_file?
                 # load_vqvae_hierarchical_model_wrapper logic: if dir provided, uses config choice.
                 # Let's override behavior by appending file if we want to be explicit.
                 # But load_vqvae_hierarchical_model_wrapper uses config['testing']['weights_file_choice'] inside model dir config.
                 # To enforce our config choice, we can pass the full file path.
                 vqvae_model_path = os.path.join(vqvae_model_path, weights_file)
        else:
             raise ValueError("VQ-VAE path not provided as argument and not found in config file.")

    print(f"Loading VQ-VAE model from {vqvae_model_path}")
    vqvae = load_vqvae_hierarchical_model_wrapper(vqvae_model_path, device)
    
    # Load Data
    x_all, _ = load_maestro(spectrograms_path, target_time_frames)
    print(f"Loaded data shape: {x_all.shape}")
    
    # Pre-calculate indices
    # Note: num_levels=2 implies we get [top_indices, bottom_indices]
    quantization_batch_size = config['training'].get('quantization_batch_size', 32)
    dataset = HierarchicalQuantizedDataset(x_all, vqvae, device, num_levels=2, batch_size=quantization_batch_size)
    
    # Split Train/Val
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize Hierarchical PixelCNN
    # Construct lists from config sections
    hidden_units = [top_cfg['hidden_channels'], bot_cfg['hidden_channels']]
    num_layers = [top_cfg['num_layers'], bot_cfg['num_layers']]
    conv_filter_size = [top_cfg['conv_filter_size'], bot_cfg['conv_filter_size']]
    dropout = [top_cfg.get('dropout_rate', 0.0), bot_cfg.get('dropout_rate', 0.0)]
    num_embeddings = [vqvae.vq_top.num_embeddings, vqvae.vq_bottom.num_embeddings]

    # Placeholders for other params (using defaults or extending config if needed)
    # The class expects lists for residual_units, attention etc. using sensible defaults or expanding config
    
    pixelcnn = HierarchicalCondGatedPixelCNN(
        num_prior_levels=2,
        input_size=[(32, 32), (64, 64)], # Placeholder, model is convolutional so this mainly checks len
        hidden_units=hidden_units,
        num_layers=num_layers,
        conv_filter_size=conv_filter_size,
        dropout=dropout,
        num_embeddings=num_embeddings,
        # Defaulting others for now as they are not in config
        residual_units=[1024, 1024], 
        attention_layers=[0, 0],
        attention_heads=[None, None],
        output_stack_layers=[20, 20],
        conditioning_stack_residual_blocks=[None, 20] 
    ).to(device)
    
    optimizer = optim.Adam(pixelcnn.parameters(), lr=config['training']['learning_rate'])
    scaler = torch.amp.GradScaler('cuda') # mixed precision scaler
    criterion = nn.CrossEntropyLoss()
    
    # Setup Save Dir
    save_dir = config['training']['save_dir']
    model_name = config['model']['name']
    run_dir = os.path.join(save_dir, "pixelcnn_hierarchical", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    
    with open(os.path.join(run_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)
        
    epochs = config['training']['epochs']
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        pixelcnn.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_indices in pbar:
            # batch_indices is a list: [top_indices, bottom_indices]
            top_indices = batch_indices[0].to(device)   # (B, H_top, W_top)
            bot_indices = batch_indices[1].to(device)   # (B, H_bot, W_bot)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                # 1. Train Top Prior
                # Forward Top
                logits_top = pixelcnn(top_indices, level='top') # (B, 256, 1, H, W)
                # Target for CrossEntropy should be (B, H, W)
                # Loss input: (B, 256, H, W) (after squeeze)
                curr_loss_top = criterion(logits_top.squeeze(2), top_indices)
                
                # 2. Train Bottom Prior
                # Forward Bottom (Conditioned on Top)
                logits_bot = pixelcnn(bot_indices, cond=top_indices, level='bottom')
                curr_loss_bot = criterion(logits_bot.squeeze(2), bot_indices)
                
                loss = curr_loss_top + curr_loss_bot
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'top': curr_loss_top.item(), 'bot': curr_loss_bot.item()})
            
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation
        pixelcnn.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch_indices in val_loader:
                top_indices = batch_indices[0].to(device)
                bot_indices = batch_indices[1].to(device)
                
                with torch.amp.autocast('cuda'):
                    logits_top = pixelcnn(top_indices, level='top')
                    loss_top = criterion(logits_top.squeeze(2), top_indices)
                    
                    logits_bot = pixelcnn(bot_indices, cond=top_indices, level='bottom')
                    loss_bot = criterion(logits_bot.squeeze(2), bot_indices)
                
                val_running_loss += (loss_top + loss_bot).item()
        
        val_loss = val_running_loss / len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state': pixelcnn.state_dict(),
                'config': config
            }, os.path.join(run_dir, "best_model.pth"))
            print("Saved Best Model")
            
        # Periodic Save
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model_state': pixelcnn.state_dict(),
            }, os.path.join(run_dir, f"model_epoch_{epoch+1}.pth"))
            
    plot_losses(train_losses, val_losses, run_dir)

if __name__ == "__main__":
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/config_pixelcnn_hierarchical.yaml', help='Path to PixelCNN config')
    parser.add_argument('--vqvae', type=str, default=None, help='Path to trained Hierarchical VQ-VAE model (overrides config)')
    args = parser.parse_args()
    
    train_pixel_cnn_hierarchical(args.config, args.vqvae)

