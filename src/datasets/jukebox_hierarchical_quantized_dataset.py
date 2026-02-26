import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np


class JukeboxHierarchicalQuantizedDataset(Dataset):
    def __init__(
        self,
        x_train: np.ndarray,
        top_model,
        middle_model,
        bottom_model,
        device: torch.device,
        batch_size: int = 32,
    ):
        """
        Pre-compute top/middle/bottom Jukebox VQ-VAE code indices for prior training.

        Args:
            x_train: Numpy array (N, H, W, C).
            top_model: Trained top-level JukeboxVQVAE.
            middle_model: Trained middle-level JukeboxVQVAE.
            bottom_model: Trained bottom-level JukeboxVQVAE.
            device: Device for quantization forward passes.
            batch_size: Batch size used during code extraction.
        """
        self.top_indices = []
        self.middle_indices = []
        self.bottom_indices = []

        for model in (top_model, middle_model, bottom_model):
            model.eval()
            model.to(device)

        print(
            f"Pre-calculating Jukebox top/middle/bottom indices for prior training "
            f"(batch_size={batch_size})..."
        )

        num_samples = len(x_train)
        with torch.no_grad():
            for i in tqdm(range(0, num_samples, batch_size)):
                batch = x_train[i:i + batch_size]
                batch_torch = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(device)

                # Top indices
                top_encoded = top_model.encoder(batch_torch)
                top_pre_vq = top_model.pre_vq_conv(top_encoded)
                _, idx_top, _, _, _ = top_model.vq(top_pre_vq)

                # Middle indices
                middle_encoded = middle_model.encoder(batch_torch)
                middle_pre_vq = middle_model.pre_vq_conv(middle_encoded)
                _, idx_middle, _, _, _ = middle_model.vq(middle_pre_vq)

                # Bottom indices
                bottom_encoded = bottom_model.encoder(batch_torch)
                bottom_pre_vq = bottom_model.pre_vq_conv(bottom_encoded)
                _, idx_bottom, _, _, _ = bottom_model.vq(bottom_pre_vq)

                self.top_indices.append(idx_top.cpu())
                self.middle_indices.append(idx_middle.cpu())
                self.bottom_indices.append(idx_bottom.cpu())

        self.top_indices = torch.cat(self.top_indices, dim=0).long()
        self.middle_indices = torch.cat(self.middle_indices, dim=0).long()
        self.bottom_indices = torch.cat(self.bottom_indices, dim=0).long()

        print(f"Top indices shape:    {tuple(self.top_indices.shape)}")
        print(f"Middle indices shape: {tuple(self.middle_indices.shape)}")
        print(f"Bottom indices shape: {tuple(self.bottom_indices.shape)}")

    def __len__(self):
        return self.top_indices.shape[0]

    def __getitem__(self, idx):
        return [
            self.top_indices[idx],
            self.middle_indices[idx],
            self.bottom_indices[idx],
        ]
