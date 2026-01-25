import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class QuantizedDataset(Dataset):
    def __init__(self, x_train, vqvae_model, device, batch_size=32):
        """
        Dataset that pre-calculates VQ-VAE indices for training a prior model (e.g. PixelCNN).
        """
        self.indices = []
        vqvae_model.eval()
        vqvae_model.to(device)
        
        print(f"Pre-calculating VQ-VAE indices for PixelCNN training (batch_size={batch_size})...")
        num_samples = len(x_train)
        
        with torch.no_grad():
            for i in tqdm(range(0, num_samples, batch_size)):
                batch = x_train[i:i+batch_size]
                # Convert to torch: (B, H, W, 1) -> (B, 1, H, W)
                batch_torch = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(device)
                
                # Get indices from VQ-VAE
                z_e = vqvae_model.encoder(batch_torch)
                _, indices, _, _, _ = vqvae_model.vq(z_e)
                
                # indices shape from VQ is (B, H_latent, W_latent)
                self.indices.append(indices.cpu())
        
        self.indices = torch.cat(self.indices, dim=0)
        print(f"Indices shape: {self.indices.shape}")
        print(f"Indices range: {self.indices.min()} - {self.indices.max()}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Return indices for a single sample
        return self.indices[idx]
