import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

from modeling.torch.vq_vae_hierarchical import VQ_VAE_Hierarchical

class HierarchicalQuantizedDataset(Dataset):
    def __init__(self, x_train: np.ndarray, vqvae_model: VQ_VAE_Hierarchical, device: torch.device, num_levels: int = 2, batch_size: int = 32):
        """
        Pre-compute hierarchical VQ-VAE indices for training a prior (e.g. PixelCNN).

        Args:
            x_train: Numpy array of training data (N, H, W, C).
            vqvae_model: Pre-trained VQ-VAE model with multiple levels.
            device: Torch device to perform computations on.
            num_levels: Number of hierarchical levels in the VQ-VAE. Currently, this
                dataset implementation supports exactly two levels (top and bottom),
                so ``num_levels`` must be set to 2.
            batch_size: Batch size used for pre-calculating indices (to avoid OOM).
        """
        assert num_levels == 2, (
            f"HierarchicalQuantizedDataset currently supports exactly 2 levels (top and bottom), "
            f"got num_levels={num_levels}."
        )

        self.hierarchical_indices = [[] for _ in range(num_levels)]
        vqvae_model.eval()
        vqvae_model.to(device)
        
        print(f"Pre-calculating hierarchical VQ-VAE indices for PixelCNN training (batch_size={batch_size})...")
        num_samples = len(x_train)
        
        with torch.no_grad():
            for i in tqdm(range(0, num_samples, batch_size)):
                batch = x_train[i:i+batch_size]
                # Convert to torch: (B, H, W, C) -> (B, C, H, W)
                batch_torch = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(device)
                
                # Get hierarchical indices from VQ-VAE
                # VQ_VAE_Hierarchical splits encoding into bottom and top steps
                enc_b = vqvae_model.encoder_bottom(batch_torch)
                enc_t = vqvae_model.encoder_top(enc_b)
                
                quant_t = vqvae_model.pre_vq_conv_top(enc_t)
                quant_b = vqvae_model.pre_vq_conv_bottom(enc_b)
                
                # VQ returns: z_q, indices, vq_loss, codebook_loss, commitment_loss
                # indices are (B*H*W, ) flattened or (B, H, W) depending on implementation
                # Let's check vector_quantizer implementation via printed shapes if needed
                # The VQ implementation returns indices_map which is (B, H, W)
                
                _, indices_t, _, _, _ = vqvae_model.vq_top(quant_t)
                _, indices_b, _, _, _ = vqvae_model.vq_bottom(quant_b)
                
                # Store indices (Top is level 0, Bottom is level 1 for PixelCNN training order)
                # indices should be (B, H, W)
                self.hierarchical_indices[0].append(indices_t.cpu())
                self.hierarchical_indices[1].append(indices_b.cpu())
        
        for level in range(num_levels):
            if len(self.hierarchical_indices[level]) > 0:
                self.hierarchical_indices[level] = torch.cat(self.hierarchical_indices[level], dim=0)
                print(f"Level {level} Indices shape (N, H, W): {self.hierarchical_indices[level].shape}")
                
                # Ensure it's long type
                self.hierarchical_indices[level] = self.hierarchical_indices[level].long()
            else:
                print(f"Level {level} Indices empty!")

    def __len__(self):
        return len(self.hierarchical_indices[0])

    def __getitem__(self, idx):
        # Return indices for all levels for a single sample
        return [self.hierarchical_indices[level][idx] for level in range(len(self.hierarchical_indices))]