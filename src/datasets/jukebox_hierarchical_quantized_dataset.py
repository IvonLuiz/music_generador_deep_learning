import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from modeling.torch.jukebox_vq_vae import JukeboxVQVAE


class JukeboxHierarchicalQuantizedDataset(Dataset):
    def __init__(
        self,
        x_train: np.ndarray,
        file_paths: list,
        top_model: JukeboxVQVAE,
        middle_model: JukeboxVQVAE,
        bottom_model: JukeboxVQVAE,
        device: torch.device,
        batch_size: int = 32,
    ):
        """
        Pre-compute top/middle/bottom Jukebox VQ-VAE code indices for prior training.

        Args:
            x_train: Numpy array (N, H, W, C).
            file_paths: List of file paths corresponding to each sample in x_train, used to extract time_id.
            top_model: Trained top-level JukeboxVQVAE.
            middle_model: Trained middle-level JukeboxVQVAE.
            bottom_model: Trained bottom-level JukeboxVQVAE.
            device: Device for quantization forward passes.
            batch_size: Batch size used during code extraction.
        """
        self.file_paths = file_paths
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

        # Transpose dim 1 (Freq) and dim 2 (Time), then make contiguous in memory.
        # This ensures that when the training script flattens the grid, 
        # it generates all frequencies for Time 0 before moving to Time 1.
        self.top_indices = torch.cat(self.top_indices, dim=0).long().transpose(1, 2).contiguous()
        self.middle_indices = torch.cat(self.middle_indices, dim=0).long().transpose(1, 2).contiguous()
        self.bottom_indices = torch.cat(self.bottom_indices, dim=0).long().transpose(1, 2).contiguous()

        print(f"Top indices shape:    {tuple(self.top_indices.shape)}")
        print(f"Middle indices shape: {tuple(self.middle_indices.shape)}")
        print(f"Bottom indices shape: {tuple(self.bottom_indices.shape)}")

    def __len__(self):
        return self.top_indices.shape[0]

    def __getitem__(self, idx):
        # Assuming we have a list of file paths in the dataset class
        file_path = self.file_paths[idx]
        
        # Extract the segmenter number from the string 
        # example: MIDI-Unprocessed_01_R1_2011_MID--AUDIO_R1-D1_02_Track02_wav.wav_segment_000.npy -> "000"
        segmenter_number_str = file_path.split('_segment_')[-1].replace('.npy', '')
        time_id = int(segmenter_number_str)

        return [
            self.top_indices[idx],
            self.middle_indices[idx],
            self.bottom_indices[idx],
            torch.tensor([time_id], dtype=torch.long),
        ]
