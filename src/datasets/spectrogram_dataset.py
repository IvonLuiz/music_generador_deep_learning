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


class LazySpectrogramDataset(Dataset):
    """
    Loads .npy files from disk on-demand to save RAM, 
    and automatically enforces strict 256x256 dimensions.
    """
    def __init__(self, file_paths, target_time_frames=256):
        self.file_paths = file_paths
        self.target_time_frames = target_time_frames

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        spec = np.load(path)
        
        # Force it to be 2D (Frequency, Time) so we can slice it easily
        if spec.ndim == 3 and spec.shape[-1] == 1:
            spec = spec[:, :, 0]
            
        # Crop or Pad the Time dimension to be exactly 256
        if spec.shape[1] > self.target_time_frames:
            spec = spec[:, :self.target_time_frames] # Crop
        elif spec.shape[1] < self.target_time_frames:
            pad_width = self.target_time_frames - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant') # Pad
            
        # Add the Channel dimension back to the front: (1, 256, 256)
        spec = spec[np.newaxis, ...]
        
        return torch.from_numpy(spec).float()

