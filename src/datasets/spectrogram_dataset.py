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
    Loads .npy files from disk on-demand to save RAM, and enforces a fixed time-axis length (target_time_frames).
    """
    def __init__(self, file_paths, target_time_frames=256, crop_strategy="random"):
        self.file_paths = file_paths
        self.target_time_frames = target_time_frames
        self.crop_strategy = str(crop_strategy).lower()
        if self.crop_strategy not in ("random", "center", "non_overlapping"):
            raise ValueError("crop_strategy must be one of: random, center, non_overlapping")
        self._fixed_windows = self._build_fixed_windows() if self.crop_strategy == "non_overlapping" else None

    def __len__(self):
        if self._fixed_windows is not None:
            return len(self._fixed_windows)
        return len(self.file_paths)

    def path_for_index(self, idx):
        if self._fixed_windows is not None:
            file_idx, _, _ = self._fixed_windows[idx]
            return self.file_paths[file_idx]
        return self.file_paths[idx]

    def __getitem__(self, idx):
        if self._fixed_windows is not None:
            file_idx, start_idx, frames = self._fixed_windows[idx]
            path = self.file_paths[file_idx]
        else:
            path = self.file_paths[idx]
            start_idx = None
            frames = None

        spec = np.load(path)
        
        # Force it to be 2D (Frequency, Time) so we can slice it easily
        if spec.ndim == 3 and spec.shape[-1] == 1:
            spec = spec[:, :, 0]

        total_frames = spec.shape[1]
        if self._fixed_windows is not None:
            spec = spec[:, start_idx : start_idx + frames]
            if spec.shape[1] < self.target_time_frames:
                pad_width = self.target_time_frames - spec.shape[1]
                spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
        elif total_frames > self.target_time_frames:
            if self.crop_strategy == "random":
                start_idx = np.random.randint(0, total_frames - self.target_time_frames + 1)
            else:
                start_idx = (total_frames - self.target_time_frames) // 2
            spec = spec[:, start_idx : start_idx + self.target_time_frames]
        else:
            # Pad if song is too short
            print(f"Warning: Spectrogram {path} has only {total_frames} time frames, which is less than target {self.target_time_frames}. Padding with zeros.")
            pad_width = self.target_time_frames - total_frames
            spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')  # pad
            
        # Add the channel dimension for PyTorch: (1, freq_bins, target_time_frames)
        spec = spec[np.newaxis, ...]
        
        return torch.from_numpy(spec).float()

    def _build_fixed_windows(self):
        """! Builds a list of fixed windows for non-overlapping cropping. Each entry is a tuple of (file_index, start_time_index, frame_count). """
        windows = []
        for file_idx, path in enumerate(self.file_paths):
            spec = np.load(path, mmap_mode='r')
            if spec.ndim == 3 and spec.shape[-1] == 1:
                total_frames = spec.shape[1]
            else:
                total_frames = spec.shape[1]
            if total_frames <= self.target_time_frames:
                starts = [0]
            else:
                starts = list(range(0, total_frames, self.target_time_frames))
            for start in starts:
                windows.append((file_idx, int(start), int(self.target_time_frames)))
        return windows
