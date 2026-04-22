import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import re
from modeling.torch.jukebox_vq_vae import JukeboxVQVAE


class JukeboxHierarchicalQuantizedDataset(Dataset):
    def __init__(
        self,
        top_model: JukeboxVQVAE,
        middle_model: JukeboxVQVAE,
        bottom_model: JukeboxVQVAE,
        device: torch.device,
        x_train: np.ndarray = None,
        file_paths: list = None,
        batch_size: int = 32,
        target_time_frames: int = 256,
    ):
        """!
        @brief Build a quantized hierarchical dataset by pre-encoding spectrograms with
        top/middle/bottom Jukebox VQ-VAE models.

        @details This precomputes discrete code indices once, so Transformer prior training
        can read token tensors directly without running VQ-VAE encoders every epoch.

        @param top_model Trained top-level JukeboxVQVAE used to extract top indices.
        @param middle_model Trained middle-level JukeboxVQVAE used to extract middle indices.
        @param bottom_model Trained bottom-level JukeboxVQVAE used to extract bottom indices.
        @param device Device used for batched VQ-VAE forward passes.
        @param x_train Optional in-memory spectrogram tensor with shape (N, H, W, C).
        @param file_paths Optional list of spectrogram .npy files loaded lazily when x_train is None.
        @param batch_size Initial precompute batch size (auto-reduced on CUDA OOM).
        @param target_time_frames Target time-axis length used when loading from files.
        """
        if x_train is None and not file_paths:
            raise ValueError("Provide either x_train or file_paths.")

        self.file_paths = list(file_paths) if file_paths is not None else [None] * int(len(x_train))
        self.top_indices = []
        self.middle_indices = []
        self.bottom_indices = []
        self.time_ids = []
        self.target_time_frames = target_time_frames

        if x_train is not None:
            num_samples = int(len(x_train))
        else:
            num_samples = int(len(self.file_paths))

        if num_samples == 0:
            raise ValueError('No samples found for JukeboxHierarchicalQuantizedDataset.')

        use_file_loader = x_train is None


        for model in (top_model, middle_model, bottom_model):
            model.eval()
            model.to(device)

        print(
            f"Pre-calculating Jukebox top/middle/bottom indices for prior training "
            f"(batch_size={batch_size})..."
        )

        current_batch_size = max(1, int(batch_size))

        def _is_cuda_oom(exc: RuntimeError) -> bool:
            msg = str(exc).lower()
            return 'out of memory' in msg and 'cuda' in msg

        with torch.no_grad():
            pbar = tqdm(total=num_samples)
            i = 0
            while i < num_samples:
                end = min(i + current_batch_size, num_samples)
                if use_file_loader:
                    batch = self._load_spectrogram_batch(self.file_paths[i:end])
                else:
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

                    self.top_indices.append(idx_top.cpu().to(torch.int32))
                    self.middle_indices.append(idx_middle.cpu().to(torch.int32))
                    self.bottom_indices.append(idx_bottom.cpu().to(torch.int32))

                    if use_file_loader:
                        batch_time_ids = [self._extract_time_id_from_path(fp) for fp in self.file_paths[i:end]]
                    else:
                        batch_time_ids = list(range(i, end))
                    self.time_ids.append(torch.tensor(batch_time_ids, dtype=torch.long))

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
        self.top_indices = torch.cat(self.top_indices, dim=0).transpose(1, 2).contiguous()
        self.middle_indices = torch.cat(self.middle_indices, dim=0).transpose(1, 2).contiguous()
        self.bottom_indices = torch.cat(self.bottom_indices, dim=0).transpose(1, 2).contiguous()
        self.time_ids = torch.cat(self.time_ids, dim=0).contiguous()

        print(f"Top indices shape:    {tuple(self.top_indices.shape)}")
        print(f"Middle indices shape: {tuple(self.middle_indices.shape)}")
        print(f"Bottom indices shape: {tuple(self.bottom_indices.shape)}")

    def _load_spectrogram_batch(self, paths: list) -> np.ndarray:
        """!
        @brief Load and normalize a batch of spectrogram files from disk.

        @details Each file is padded or cropped to `self.target_time_frames` on the
        time axis, cast to float32, and stacked into shape (B, H, W, 1).

        @param paths List of .npy spectrogram file paths.
        @return Numpy array of shape (B, H, target_time_frames, 1).
        """
        specs = []
        for file_path in paths:
            spectrogram = np.load(file_path)
            if spectrogram.shape[1] > self.target_time_frames:
                spectrogram = spectrogram[:, :self.target_time_frames]
                print(f"Warning: spectrogram {file_path} has more time frames ({spectrogram.shape[1]}) than target ({self.target_time_frames}). Cropping.")
            elif spectrogram.shape[1] < self.target_time_frames:
                pad_width = self.target_time_frames - spectrogram.shape[1]
                spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
            specs.append(spectrogram.astype(np.float32, copy=False))

        batch_np = np.stack(specs, axis=0)[..., np.newaxis]
        return batch_np

    @staticmethod
    def _extract_time_id_from_path(file_path: str) -> int:
        """!
        @brief Extract segment index from a spectrogram filename.

        @param file_path Path expected to end with `_segment_<int>.npy`.
        @return Integer segment id parsed from filename.
        @throws ValueError If the expected naming pattern is not found.
        """
        match = re.search(r'_segment_(\d+)\.npy$', str(file_path))
        if match is None:
            raise ValueError(
                f"Could not parse segment index from file path: {file_path}. "
                "Expected suffix '_segment_<int>.npy'."
            )
        return int(match.group(1))

    def __len__(self):
        """!
        @brief Return number of precomputed samples.

        @return Dataset length.
        """
        return self.top_indices.shape[0]

    def __getitem__(self, idx):
        """!
        @brief Fetch one training sample with hierarchical token targets and time id.

        @param idx Sample index.
        @return List containing [top_indices, middle_indices, bottom_indices, time_id].
        """
        return [
            self.top_indices[idx].long(),
            self.middle_indices[idx].long(),
            self.bottom_indices[idx].long(),
            self.time_ids[idx].view(1),
        ]
