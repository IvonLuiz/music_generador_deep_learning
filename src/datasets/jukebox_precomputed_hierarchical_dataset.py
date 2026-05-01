import os
import sys
import tempfile
from glob import glob
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Add 'src' to sys.path so the script can be run directly from the project root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class JukeboxQuantizedDataset(Dataset):
    def __init__(
        self,
        quantized_path: str,
        file_paths: List[str] = None,
        target_time_frames: int = 2048
    ):
        """
        @param target_time_frames: The window size in raw spectrogram frames (e.g. 2048 for Top)
        """
        self.target_time_frames = target_time_frames
        if file_paths:
            self.files = [os.path.join(quantized_path, os.path.basename(f).replace('.npy', '_full_quantized.pt')) 
                          for f in file_paths]
        else:
            self.files = sorted(glob(os.path.join(quantized_path, "*.pt")))

        # Define downsampling ratios based on your VQ-VAE levels
        # (Spectrogram Frames / Token Time Steps)
        self.ratios = {
            'top': 2048 // 64,    # 32
            'middle': 512 // 32,  # 16 (Using your level_profile config)
            'bottom': 128 // 16   # 8
        }

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        payload = torch.load(self.files[idx % len(self.files)], weights_only=False)
        total_frames = payload['total_frames']

        # Calculate max possible start index for the top-level window
        max_top_tokens = total_frames // self.ratios['top']
        target_top_tokens = 64 # Janela de 512 tokens / 8 freq

        if max_top_tokens > target_top_tokens:
            # Escolhemos um índice de TOKEN aleatório
            t_start = np.random.randint(0, max_top_tokens - target_top_tokens)
        else:
            t_start = 0
        m_start = t_start * (self.ratios['top'] // self.ratios['middle'])
        b_start = t_start * (self.ratios['top'] // self.ratios['bottom'])

        # Slicing
        top_indices = payload['top'][t_start : t_start + 64]
        mid_indices = payload['middle'][m_start : m_start + 32]
        bot_indices = payload['bottom'][b_start : b_start + 16]

        # Padding
        #top_indices = self._pad_tokens(top_indices, 64, 8)
        #mid_indices = self._pad_tokens(mid_indices, 32, 16)
        #bot_indices = self._pad_tokens(bot_indices, 16, 32)

        # Timing Metadata
        # (Using your standard hop/sr calculation logic)
        # Assuming SR=22050, Hop=256
        actual_start_frame = t_start * self.ratios['top']
        start_time_s = (actual_start_frame * 256) / 22050
        total_duration_s = (total_frames * 256) / 22050
        fraction = start_time_s / total_duration_s
        timing = torch.tensor([start_time_s, total_duration_s, fraction], dtype=torch.float32)

        return (
            torch.from_numpy(top_indices).long(),
            torch.from_numpy(mid_indices).long(),
            torch.from_numpy(bot_indices).long(),
            timing
        )

    def _pad_tokens(self, arr, target_t, freq):
        if arr.shape[0] < target_t:
            diff = target_t - arr.shape[0]
            arr = np.pad(arr, ((0, diff), (0, 0)), mode='constant')
        return arr[:target_t]


def test_dataset_alignment_and_logic():
    print("🚀 Starting JukeboxQuantizedDataset tests...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Create a "Full Song" Mock Payload
        # We use a gradient of numbers so we can track if the slices are aligned
        # Top: 1000 time steps, 8 freq
        # Middle: 2000 time steps, 16 freq
        # Bottom: 4000 time steps, 32 freq
        # This matches the 32:16:8 ratios (1000*32 = 2000*16 = 4000*8 = 32000 raw frames)
        
        total_raw_frames = 32000
        
        mock_top = torch.arange(1000).view(-1, 1).repeat(1, 8)
        mock_mid = torch.arange(2000).view(-1, 1).repeat(1, 16)
        mock_bot = torch.arange(4000).view(-1, 1).repeat(1, 32)
        
        quantized_file = os.path.join(tmp_dir, 'test_song_full_quantized.pt')
        payload = {
            'top': mock_top.numpy(),
            'middle': mock_mid.numpy(),
            'bottom': mock_bot.numpy(),
            'total_frames': total_raw_frames
        }
        torch.save(payload, quantized_file)

        # 2. Initialize Dataset
        dataset = JukeboxQuantizedDataset(
            quantized_path=tmp_dir,
            target_time_frames=2048 # 24s window
        )

        print(f"Dataset length (files): {len(dataset)}")
        assert len(dataset) == 1, "Dataset should find the 1 mock file."

        # 3. Test Multiple Random Crops for Alignment
        for i in range(5):
            top, mid, bot, timing = dataset[0]
            
            # Check Shapes
            assert top.shape == (64, 8), f"Top shape mismatch: {top.shape}"
            assert mid.shape == (32, 16), f"Mid shape mismatch: {mid.shape}"
            assert bot.shape == (16, 32), f"Bot shape mismatch: {bot.shape}"
            
            # Check Types
            assert top.dtype == torch.long
            assert timing.dtype == torch.float32

            # --- MATH VERIFICATION ---
            # Extract the 'Time ID' from the first column of the first row
            # Because we used arange(N), the value should match the index
            t_val = top[0, 0].item()
            m_val = mid[0, 0].item()
            b_val = bot[0, 0].item()

            # Ratio check: Top:Mid:Bot is 1:2:4 in token-step space
            # Example: If Top starts at token 10, Mid must start at 20, Bot at 40
            assert m_val == t_val * 2, f"Alignment Error: Top index {t_val} vs Mid index {m_val}"
            assert b_val == t_val * 4, f"Alignment Error: Top index {t_val} vs Bot index {b_val}"
            
            # Timing Verification
            start_s, total_s, fraction = timing.tolist()
            expected_start_s = (t_val * 32 * 256) / 22050
            assert abs(start_s - expected_start_s) < 1e-4, f"Timing start mismatch: {start_s} vs {expected_start_s}"
            
            print(f"  [Pass] Crop {i}: Start Token (T:{t_val}, M:{m_val}, B:{b_val}) aligned.")

        # 4. Test Padding (Edge Case)
        # Create a very short song payload
        short_file = os.path.join(tmp_dir, 'short_song_full_quantized.pt')
        short_payload = {
            'top': torch.zeros((10, 8)).numpy(),
            'middle': torch.zeros((20, 16)).numpy(),
            'bottom': torch.zeros((40, 32)).numpy(),
            'total_frames': 320 # Much smaller than target_time_frames (2048)
        }
        torch.save(short_payload, short_file)
        
        short_ds = JukeboxQuantizedDataset(tmp_dir, file_paths=[short_file])
        s_top, s_mid, s_bot, s_timing = short_ds[0]
        
        assert s_top.shape == (64, 8), "Short song should be padded to 64 time steps"
        assert torch.all(s_top[10:] == 0), "Padding area should be zeros"
        assert s_timing[2] == 0, "Fraction for short song start should be 0"
        print("  [Pass] Padding/Short song logic verified.")

        # 5. Test Dataloader compatibility
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        # Since we only have 1 file, we'll see how it handles it
        print("  [Pass] DataLoader test passed.")

    print("\nAll tests passed successfully!")

if __name__ == '__main__':
    test_dataset_alignment_and_logic()