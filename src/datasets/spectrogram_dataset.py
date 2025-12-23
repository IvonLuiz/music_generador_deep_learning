import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    def __init__(self, x: np.ndarray):
        # Expect (N, H, W, 1) with values in [0,1]
        assert x.ndim == 4 and x.shape[-1] == 1
        self.x = x.astype(np.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        spec = self.x[idx]  # (H, W, 1)
        # To torch (C,H,W)
        spec = np.transpose(spec, (2, 0, 1))  # (1, H, W)
        return torch.from_numpy(spec)


class MmapSpectrogramDataset(Dataset):
    def __init__(self, mmap_array, indices=None):
        """
        Dataset that reads from a memory-mapped array using specific indices.
        Avoids loading the entire dataset into RAM.
        
        Args:
            mmap_array: The numpy memory-mapped array (N, H, W, C).
            indices: Array of indices to use for this dataset. If None, uses all.
        """
        self.mmap_array = mmap_array
        self.indices = indices if indices is not None else np.arange(len(mmap_array))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        # Read specific sample from mmap array
        spec = self.mmap_array[real_idx]  # (H, W, 1)
        
        # Ensure float32 and correct shape
        spec = spec.astype(np.float32)
        spec = np.transpose(spec, (2, 0, 1))  # (1, H, W)
        return torch.from_numpy(spec)
