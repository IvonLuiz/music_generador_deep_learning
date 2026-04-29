import os
import re
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Add 'src' to sys.path so the script can be run directly from the project root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling.torch.jukebox_vq_vae import JukeboxVQVAE
from train_scripts.jukebox_utils import extract_song_prefix


class JukeboxHierarchicalQuantizedDataset(Dataset):
    def __init__(
        self,
        top_model: JukeboxVQVAE,
        middle_model: JukeboxVQVAE,
        bottom_model: JukeboxVQVAE,
        device: torch.device,
        x_train: np.ndarray = None,
        file_paths: list = None,
        batch_size: int = 1,
        target_time_frames: int = 2048,
        level_target_time_frames: dict = {},
        sample_rate: int = 22050,
        hop_length: int = 256,
        segment_overlap: float = 0.0,
        target_level: str = 'top',
    ):
        """!
        @brief Build a quantized hierarchical dataset by pre-encoding spectrograms with
        top/middle/bottom Jukebox VQ-VAE models using level-specific temporal windows.

        @details Each level encodes a different-length slice of the same spectrogram, so
        every level produces the same number of tokens while covering different audio durations
        (Jukebox §5.2 hierarchical context). Timing metadata [start_time_s, total_duration_s,
        fraction_elapsed] is attached per sample so the Transformer can learn global position.

        Spectrogram files must follow the naming convention: <prefix>_segment_<int>.npy

        @param top_model Trained top-level JukeboxVQVAE.
        @param middle_model Trained middle-level JukeboxVQVAE.
        @param bottom_model Trained bottom-level JukeboxVQVAE.
        @param device Device for batched VQ-VAE forward passes.
        @param x_train Optional in-memory spectrogram array (N, H, W, C).
        @param file_paths Optional list of .npy spectrogram paths (lazy loading).
        @param batch_size Initial pre-compute batch size (auto-reduced on CUDA OOM).
        @param target_time_frames Time frames to load from disk (= Top level's window).
        @param level_target_time_frames Per-level encoding windows, e.g.
               {'top': 2048, 'middle': 512, 'bottom': 128}. Keys not present fall back
               to target_time_frames.
        @param sample_rate Audio sample rate used during preprocessing (default 22050).
        @param hop_length STFT hop length used during preprocessing (default 256).
        @param segment_overlap Overlap fraction used during preprocessing (default 0.0).
        """
        if x_train is None and not file_paths:
            raise ValueError("Provide either x_train or file_paths.")

        self.device = device
        self.target_time_frames = target_time_frames
        self.file_paths = list(file_paths) if file_paths is not None else [None] * int(len(x_train))

        self.levels = ['top', 'middle', 'bottom']
        self.models = {
            'top': top_model,
            'middle': middle_model,
            'bottom': bottom_model
        }

        self.time_frames = {
            lvl: int(level_target_time_frames.get(lvl, target_time_frames))
            for lvl in self.levels
        }

        self.indices = {lvl: [] for lvl in self.levels}
        num_samples = int(len(x_train)) if x_train is not None else int(len(self.file_paths))
        if num_samples == 0:
            raise ValueError('No samples found for JukeboxHierarchicalQuantizedDataset.')

        for model in self.models.values():
            model.eval().to(self.device)
            
        self._precompute_timing_indices(
            target_time_frames, sample_rate, hop_length, segment_overlap
        )

        self._compute_indices_with_auto_batching(
            x_train, batch_size, num_samples
        )

    def _precompute_timing_indices(
        self,
        target_time_frames,
        sample_rate,
        hop_length,
        segment_overlap
    ):
        """!
        Pre-compute timing indices for each sample.
        
        @details Timing is derived from the FULL loaded segment length (target_time_frames)
        so it represents global song position, independent of which level is being trained.
        It is calculated based on the segment index extracted from the file name and the
        total number of segments for that song, which is determined by finding the maximum
        segment index for each song prefix across all file paths.
        The timing information is stored as a tensor of shape (N, 3) where each row contains 
        [start_time_s, total_duration_s, fraction_elapsed] for the corresponding sample.
        
        @param target_time_frames Number of time frames in the full segment.
        @param sample_rate Sample rate of the original audio.
        @param hop_length Hop length used in the STFT during preprocessing).
        @param segment_overlap Overlap fraction used during preprocessing.
        """
        # Pre-compute per-segment timing [start_time_s, total_duration_s, fraction_elapsed].
        # Timing is derived from the FULL loaded segment length (target_time_frames) so it
        # represents global song position, independent of which level is being trained.
        segment_samples = max(1, (target_time_frames - 1) * hop_length)
        segment_duration_s = segment_samples / sample_rate
        hop_samples = max(1, int(segment_samples * (1.0 - float(segment_overlap))))
        hop_duration_s = hop_samples / sample_rate

        raw_timing_info = [
            (self._extract_time_id_from_path(fp), extract_song_prefix(fp)) 
            if fp is not None else (0, "__none__") 
            for fp in self.file_paths
        ]
        seg_indices = np.array([x[0] for x in raw_timing_info])
        prefixes = np.array([x[1] for x in raw_timing_info])

        # Map each unique prefix to its max segment index
        unique_prefixes, inverse_indices = np.unique(prefixes, return_inverse=True)
        max_indices_per_prefix = np.zeros_like(unique_prefixes, dtype=int)
        np.maximum.at(max_indices_per_prefix, inverse_indices, seg_indices)
        num_segs_per_sample = max_indices_per_prefix[inverse_indices] + 1
        
        # Calculate timing matrix
        start_times = seg_indices * hop_duration_s
        total_durations = (num_segs_per_sample - 1) * hop_duration_s + segment_duration_s
        denominators = np.maximum(num_segs_per_sample - 1, 1)   # avoiding div by zero
        fractions = seg_indices / denominators

        # Handle the "__none__" special case
        none_mask = (prefixes == "__none__")
        start_times[none_mask] = 0.0
        total_durations[none_mask] = segment_duration_s
        fractions[none_mask] = 0.0
        
        # Combine into (N, 3) tensor
        timing_np = np.stack([start_times, total_durations, fractions], axis=1)
        self.timing = torch.from_numpy(timing_np).float()

    def _compute_indices_with_auto_batching(
        self,
        x_train,
        batch_size,
        num_samples
    ):
        """!
        Compute VQ-VAE indices for a batch, automatically reducing batch size on CUDA OOM.
        
        @param model JukeboxVQVAE model to use for encoding and quantization.
        @param batch Input batch tensor (B, C, H, W).
        @param device Device to perform computations on.
        @return Quantized indices tensor (B, H, W) on CPU.
        """
        current_batch_size = max(1, int(batch_size))
        use_file_loader = x_train is None

        print(
            f"Pre-calculating Jukebox top/middle/bottom indices for prior training "
            f"(batch_size={batch_size}, frames={[self.time_frames[level] for level in self.levels]})..."
        )

        current_batch_size = max(1, int(batch_size))

        with torch.no_grad():
            pbar = tqdm(total=num_samples)
            i = 0
            while i < num_samples:
                end = min(i + current_batch_size, num_samples)
                try:
                    batch_np = self._load_spectrogram_batch(self.file_paths[i:end]) if use_file_loader else x_train[i:end]

                    # Convert to torch: (B, 1, Freq, Time)
                    full_torch = torch.from_numpy(batch_np).permute(0, 3, 1, 2).float().to(self.device)

                    # Unified Encoding Loop
                    for lvl in self.levels:
                        # Slice to level window
                        sliced_batch = full_torch[:, :, :, :self.time_frames[lvl]]
                        
                        # Forward pass
                        model = self.models[lvl]
                        encoded = model.encoder(sliced_batch)
                        pre_vq = model.pre_vq_conv(encoded)
                        _, idx, _, _, _ = model.vq(pre_vq)

                        self.indices[lvl].append(idx.cpu().to(torch.int32))

                    pbar.update(end - i)
                    i = end

                # Catch CUDA OOM and reduce batch size, retrying the same batch
                except RuntimeError as exc:
                    if 'out of memory' in str(exc).lower() and current_batch_size > 1:
                        if self.device.type != 'cuda' or current_batch_size == 1:
                            print("Reached minimum batch size or not using CUDA, cannot reduce further. Raising exception.")
                            pbar.close()
                            raise
                        
                        new_batch_size = max(1, current_batch_size // 2)
                        print(
                            f"CUDA OOM while precomputing indices at batch_size={current_batch_size}. "
                            f"Retrying with batch_size={new_batch_size}."
                        )
                        current_batch_size = new_batch_size
                        torch.cuda.empty_cache()

                        continue

            pbar.close()

        # Concatenate and reformat for all levels
        for lvl in self.levels:
            # Transpose dim 1 (Freq) and dim 2 (Time) → store as (N, Time, Freq).
            # When the training script flattens (Time, Freq) row-major, Time is the outer
            # loop so every block_len consecutive tokens span one time step of the 2D grid.
            self.indices[lvl] = torch.cat(self.indices[lvl], dim=0).transpose(1, 2).contiguous()
            print(f"{lvl.capitalize()} indices shape: {tuple(self.indices[lvl].shape)}")

    def _load_spectrogram_batch(self, paths: list) -> np.ndarray:
        """!
        @brief Load and pad/crop a batch of spectrogram files to target_time_frames.

        @details Sliding window logic: If a spectrogram has more time frames than 
        target_time_frames, we randomly select a contiguous slice of length target_time_frames.
        If it has fewer, we pad with zeros on the right.

        @param paths List of .npy spectrogram file paths.
        @return Numpy array of shape (B, H, target_time_frames, 1).
        """
        specs = []
        for file_path in paths:
            spectrogram = np.load(file_path)
            
            # Handling files longer than target
            if spectrogram.shape[1] > self.target_time_frames:
                max_start = spectrogram.shape[1] - self.target_time_frames
                start = np.random.randint(0, max_start)
                spectrogram = spectrogram[:, start : start + self.target_time_frames]
            
            # Handling files shorter than target
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
        return self.timing.shape[0] # all levels have the same number of samples

    def __getitem__(self, idx):
        """!
        @brief Fetch one training sample.

        @return [top_indices, middle_indices, bottom_indices, timing] where timing is
                (3,) float32: [start_time_s, total_duration_s, fraction_elapsed].
        """
        sample = [self.indices[lvl][idx].long() for lvl in self.levels]
        sample.append(self.timing[idx])
        return sample

    @property
    def top_indices(self):
        return self.indices.get('top')

    @property
    def middle_indices(self):
        return self.indices.get('middle')

    @property
    def bottom_indices(self):
        return self.indices.get('bottom')


if __name__ == "__main__":
    # Example usage (requires actual VQ-VAE models and spectrogram files):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    top_model = JukeboxVQVAE(hidden_dim=1024, levels=3, input_channels=1)
    middle_model = JukeboxVQVAE(hidden_dim=1024, levels=2, input_channels=1)
    bottom_model = JukeboxVQVAE(hidden_dim=1024, levels=1, input_channels=1)
    dataset = JukeboxHierarchicalQuantizedDataset(
        top_model=top_model,
        middle_model=middle_model,
        bottom_model=bottom_model,
        device=device,
        file_paths=[
            'data/processed/maestro/MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_01_Track01_wav.wav_segment_000.npy',
            'data/processed/maestro/MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_01_Track01_wav.wav_segment_005.npy',
            'data/processed/maestro/MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_01_Track01_wav.wav_segment_006.npy',
        ],
        target_time_frames=2048,
        level_target_time_frames={'top': 2048, 'middle': 512, 'bottom': 128},
        sample_rate=22050,
        hop_length=256,
        segment_overlap=0.5,
        target_level='top'
    )

    print(f"Dataset length: {len(dataset)}")
    sample_1 = dataset[0]
    sample_2 = dataset[1]
    sample_3 = dataset[2]
    print(f"Sample shapes: {[s.shape for s in sample_1[:-1]]}, Timing shape: {sample_1[-1].shape}")
    
    # Test timing
    print(f"Timing info sample 1 (start_time_s, total_duration_s, fraction_elapsed): {sample_1[-1].tolist()}")
    print(f"Timing info sample 2  (start_time_s, total_duration_s, fraction_elapsed): {sample_2[-1].tolist()}")
    print(f"Timing info sample 3 (start_time_s, total_duration_s, fraction_elapsed): {sample_3[-1].tolist()}")
    assert sample_1[-1].shape == (3,), "Timing info should have shape (3,)"
    assert sample_2[-1][1] == sample_1[-1][1], "Total durations should match for samples from the same song"
    assert sample_1[-1][0] == 0.0, "First sample should start at 0s"
    assert sample_2[-1][0] > 0.0, "Second sample should have a positive start time"
    assert sample_1[-1][2] == 0.0, "First sample should have fraction_elapsed 0.0"
    assert sample_2[-1][2] > 0.0, "Second sample should have fraction_elapsed > 0.0"
    assert sample_3[-1][2] == 1.0, "Third sample should have fraction_elapsed 1.0 (last segment of the song)"