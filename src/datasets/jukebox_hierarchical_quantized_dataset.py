import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import re
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
        current_batch_size = max(1, int(batch_size))

        def _is_cuda_oom(exc: RuntimeError) -> bool:
            msg = str(exc).lower()
            return 'out of memory' in msg and 'cuda' in msg

        with torch.no_grad():
            pbar = tqdm(total=num_samples)
            i = 0
            while i < num_samples:
                end = min(i + current_batch_size, num_samples)
                batch = x_train[i:end]

                try:
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

                    # Release transient CUDA tensors before next chunk
                    del batch_torch
                    del top_encoded, top_pre_vq, idx_top
                    del middle_encoded, middle_pre_vq, idx_middle
                    del bottom_encoded, bottom_pre_vq, idx_bottom

                    pbar.update(end - i)
                    i = end

                except RuntimeError as exc:
                    if not _is_cuda_oom(exc):
                        pbar.close()
                        raise

                    if device.type != 'cuda' or current_batch_size == 1:
                        pbar.close()
                        raise

                    new_batch_size = max(1, current_batch_size // 2)
                    print(
                        f"CUDA OOM while precomputing indices at batch_size={current_batch_size}. "
                        f"Retrying with batch_size={new_batch_size}."
                    )
                    current_batch_size = new_batch_size
                    torch.cuda.empty_cache()

            pbar.close()

        # Transpose dim 1 (Freq) and dim 2 (Time), then make contiguous in memory.
        # This ensures that when the training script flattens the grid, 
        # it generates all frequencies for Time 0 before moving to Time 1.
        self.top_indices = torch.cat(self.top_indices, dim=0).long().transpose(1, 2).contiguous()
        self.middle_indices = torch.cat(self.middle_indices, dim=0).long().transpose(1, 2).contiguous()
        self.bottom_indices = torch.cat(self.bottom_indices, dim=0).long().transpose(1, 2).contiguous()

        print(f"Top indices shape:    {tuple(self.top_indices.shape)}")
        print(f"Middle indices shape: {tuple(self.middle_indices.shape)}")
        print(f"Bottom indices shape: {tuple(self.bottom_indices.shape)}")

    @staticmethod
    def _extract_time_id_from_path(file_path: str) -> int:
        match = re.search(r'_segment_(\d+)\.npy$', str(file_path))
        if match is None:
            raise ValueError(
                f"Could not parse segment index from file path: {file_path}. "
                "Expected suffix '_segment_<int>.npy'."
            )
        return int(match.group(1))

    def __len__(self):
        return self.top_indices.shape[0]

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        # Extract the segmenter number from the string 
        # example: MIDI-Unprocessed_01_R1_2011_MID--AUDIO_R1-D1_02_Track02_wav.wav_segment_000.npy -> "000"
        time_id = self._extract_time_id_from_path(file_path)

        return [
            self.top_indices[idx],
            self.middle_indices[idx],
            self.bottom_indices[idx],
            torch.tensor([time_id], dtype=torch.long),
        ]
