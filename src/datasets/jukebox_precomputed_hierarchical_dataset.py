import json
import os
import sys
import tempfile
from glob import glob
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Add 'src' to sys.path so the script can be run directly from the project root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class JukeboxQuantizedDataset(Dataset):
    """
    Dataset for hierarchical transformer prior training.

    Returns (target, cond, second_cond, timing) for every level:
      - top:    target=top,    cond=None,              second_cond=None
      - middle: target=middle, cond=aligned_top_slice, second_cond=None
      - bottom: target=bottom, cond=aligned_mid_slice, second_cond=aligned_top_slice

    The conditioning slices cover the SAME audio span as the target window,
    which is the correct alignment per the Jukebox paper.
    """

    def __init__(
        self,
        quantized_path: str,
        file_paths: List[str] = None,
        target_time_frames: int = 2048,
        level_target_time_frames: Optional[Dict[str, int]] = None,
        selected_level: str = 'top',
        sample_rate: int = 22050,
        hop_length: int = 256,
    ):
        """
        Args:
            quantized_path:           Directory containing *_full_quantized.pt files.
            file_paths:               Optional list of source .npy paths; mapped to .pt names.
            target_time_frames:       Top-level window size in raw spectrogram frames.
            level_target_time_frames: Dict with per-level window sizes, e.g.
                                      {'top': 2048, 'middle': 512, 'bottom': 128}.
            selected_level:           Which prior is being trained ('top'/'middle'/'bottom').
            sample_rate:              Audio sample rate (used for timing metadata).
            hop_length:               STFT hop length (used for timing metadata).
        """
        self.selected_level = selected_level
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.quantized_path = quantized_path
        self.mode = 'legacy_full_song'
        self.window_entries = []

        lvl = level_target_time_frames or {}
        self.top_tf    = int(lvl.get('top',    target_time_frames))
        self.middle_tf = int(lvl.get('middle', target_time_frames))
        self.bottom_tf = int(lvl.get('bottom', target_time_frames))

        if not self._init_windowed_files(file_paths=file_paths):
            if file_paths:
                self.files = [
                    os.path.join(quantized_path, os.path.basename(f).replace('.npy', '_full_quantized.pt'))
                    for f in file_paths
                ]
            else:
                self.files = sorted(glob(os.path.join(quantized_path, "*.pt")))

        # Peek at the first file to learn the actual token grid shapes.
        # This replaces all hardcoded ratio constants.
        self._init_grids()

    def _init_windowed_files(self, file_paths: Optional[List[str]]) -> bool:
        manifest_path = os.path.join(self.quantized_path, 'windowed_manifest.jsonl')
        if not os.path.isfile(manifest_path):
            return False

        allowed_stems = None
        if file_paths:
            allowed_stems = {self._source_stem(path) for path in file_paths}

        entries = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if allowed_stems is not None and record.get('source_stem') not in allowed_stems:
                    continue
                if self.selected_level not in record.get('eligible_levels', []):
                    continue
                file_path = os.path.join(self.quantized_path, record['file'])
                if not os.path.isfile(file_path):
                    continue
                entries.append(record)

        if not entries:
            return False

        self.mode = 'windowed'
        self.window_entries = entries
        self.files = [os.path.join(self.quantized_path, entry['file']) for entry in entries]
        return True

    def _init_grids(self):
        """Read one payload to discover both full-song and fixed-window token grids."""
        if not self.files:
            raise ValueError("No quantized .pt files found.")
        payload = torch.load(self.files[0], weights_only=False)
        if payload.get('format') == 'windowed_v1':
            self._init_windowed_grids(payload)
            return

        self._init_legacy_grids(payload)

    def _init_windowed_grids(self, payload: dict):
        self.mode = 'windowed'
        self.top_grid = tuple(int(x) for x in self._shape_2d(payload['top']))
        self.middle_grid = tuple(int(x) for x in self._shape_2d(payload['middle']))
        self.bottom_grid = tuple(int(x) for x in self._shape_2d(payload['bottom']))
        self.top_full_grid = self.top_grid
        self.middle_full_grid = self.middle_grid
        self.bottom_full_grid = self.bottom_grid

        self.ratios = {
            'top': max(1, self.top_tf // self.top_grid[0]),
            'middle': max(1, self.middle_tf // self.middle_grid[0]),
            'bottom': max(1, self.bottom_tf // self.bottom_grid[0]),
        }

        self._top_window_cols = self.top_grid[0]
        self._middle_window_cols = self.middle_grid[0]
        self._bottom_window_cols = self.bottom_grid[0]
        self._top_cols_for_middle = max(1, round(self._top_window_cols * self.middle_tf / self.top_tf))
        self._top_cols_for_bottom = max(1, round(self._top_window_cols * self.bottom_tf / self.top_tf))
        self._mid_cols_for_bottom = max(1, round(self._middle_window_cols * self.bottom_tf / self.middle_tf))
        print(
            f"[JukeboxQuantizedDataset] Windowed grids — "
            f"top={self.top_grid}, middle={self.middle_grid}, bottom={self.bottom_grid} | "
            f"ratios={self.ratios} | files={len(self.files)}"
        )

    def _init_legacy_grids(self, payload: dict):
        # Each level array in the precomputed payload has shape [Total_Time_Steps, Freq_Bins].
        # We keep those full-song shapes for diagnostics, but training uses fixed-size windows.
        self.top_full_grid    = (int(payload['top'].shape[0]),    int(payload['top'].shape[1]))
        self.middle_full_grid = (int(payload['middle'].shape[0]), int(payload['middle'].shape[1]))
        self.bottom_full_grid = (int(payload['bottom'].shape[0]), int(payload['bottom'].shape[1]))

        total_frames = int(payload['total_frames'])
        # Derive downsampling ratios from actual data instead of hardcoding.
        # ratio = raw_frames / token_time_steps
        self.ratios = {
            'top':    total_frames // self.top_full_grid[0],
            'middle': total_frames // self.middle_full_grid[0],
            'bottom': total_frames // self.bottom_full_grid[0],
        }

        # Precompute how many token time-cols each level window occupies.
        # These are used to pick the right-sized conditioning slices.
        self._top_window_cols    = self.top_tf    // self.ratios['top']
        self._middle_window_cols = self.middle_tf // self.ratios['middle']
        self._bottom_window_cols = self.bottom_tf // self.ratios['bottom']

        # Expose the fixed-size training-window grids that the prior actually sees.
        self.top_grid    = (self._top_window_cols, self.top_full_grid[1])
        self.middle_grid = (self._middle_window_cols, self.middle_full_grid[1])
        self.bottom_grid = (self._bottom_window_cols, self.bottom_full_grid[1])
        print(
            f"[JukeboxQuantizedDataset] Full grids — "
            f"top={self.top_full_grid}, middle={self.middle_full_grid}, bottom={self.bottom_full_grid} | "
            f"window_grids=top={self.top_grid}, middle={self.middle_grid}, bottom={self.bottom_grid} | "
            f"ratios={self.ratios}"
        )

        # Aligned conditioning slice sizes
        # For middle trained on a middle-window, only the TOP tokens covering
        # the same audio span are used as conditioning (not the full top window).
        self._top_cols_for_middle = max(1, round(self._top_window_cols * self.middle_tf / self.top_tf))
        self._top_cols_for_bottom = max(1, round(self._top_window_cols * self.bottom_tf / self.top_tf))
        self._mid_cols_for_bottom = max(1, round(self._middle_window_cols * self.bottom_tf / self.middle_tf))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.mode == 'windowed':
            return self._getitem_windowed(idx)

        return self._getitem_legacy(idx)

    def _getitem_windowed(self, idx):
        payload = torch.load(self.files[idx % len(self.files)], weights_only=False)
        top_tensor = self._to_long_tensor(payload['top'])
        middle_tensor = self._to_long_tensor(payload['middle'])
        bottom_tensor = self._to_long_tensor(payload['bottom'])

        top_for_middle = top_tensor[:self._top_cols_for_middle]
        top_for_bottom = top_tensor[:self._top_cols_for_bottom]
        mid_for_bottom = middle_tensor[:self._mid_cols_for_bottom]

        if 'timing' in payload:
            timing = self._to_float_tensor(payload['timing'])
        else:
            start_frame = int(payload.get('start_frame', 0))
            total_frames = int(payload.get('total_frames', self.top_tf))
            start_time_s = (start_frame * self.hop_length) / self.sample_rate
            total_duration_s = (total_frames * self.hop_length) / self.sample_rate
            fraction = start_time_s / max(total_duration_s, 1e-6)
            timing = torch.tensor([start_time_s, total_duration_s, fraction], dtype=torch.float32)

        if self.selected_level == 'top':
            target = top_tensor
            cond = torch.empty(0, dtype=torch.long)
            second_cond = torch.empty(0, dtype=torch.long)
        elif self.selected_level == 'middle':
            target = middle_tensor
            cond = top_for_middle
            second_cond = torch.empty(0, dtype=torch.long)
        elif self.selected_level == 'bottom':
            target = bottom_tensor
            cond = mid_for_bottom
            second_cond = top_for_bottom
        else:
            raise ValueError(f'Unsupported selected_level: {self.selected_level}')

        return target, cond, second_cond, timing

    def _getitem_legacy(self, idx):
        payload = torch.load(self.files[idx % len(self.files)], weights_only=False)
        total_frames = int(payload['total_frames'])

        top_arr    = payload['top']     # [T_top, F_top]
        middle_arr = payload['middle']  # [T_mid, F_mid]
        bottom_arr = payload['bottom']  # [T_bot, F_bot]

        # ── Random crop anchored at the TOP level ──────────────────────────────
        # We choose a random top-token start, then derive aligned starts for
        # middle and bottom using the ratio relationships.
        max_top_start = max(0, top_arr.shape[0] - self._top_window_cols)
        t_start = int(np.random.randint(0, max_top_start + 1)) if max_top_start > 0 else 0

        # Scale to middle and bottom token indices
        top_to_mid_ratio = self.ratios['top'] // self.ratios['middle']   # e.g. 32//16 = 2
        top_to_bot_ratio = self.ratios['top'] // self.ratios['bottom']   # e.g. 32//8  = 4
        m_start = t_start * top_to_mid_ratio
        b_start = t_start * top_to_bot_ratio

        # ── Slice target window ────────────────────────────────────────────────
        top_slice    = top_arr[t_start : t_start + self._top_window_cols]
        middle_slice = middle_arr[m_start : m_start + self._middle_window_cols]
        bottom_slice = bottom_arr[b_start : b_start + self._bottom_window_cols]

        # Pad if the song is shorter than the requested window
        top_slice    = self._pad(top_slice,    self._top_window_cols)
        middle_slice = self._pad(middle_slice, self._middle_window_cols)
        bottom_slice = self._pad(bottom_slice, self._bottom_window_cols)

        # ── Aligned conditioning slices ──────────────────────────────────
        # Each conditioning slice covers the SAME audio span as the target window.
        # For middle prior: use only the top tokens aligned to the middle window span.
        # For bottom prior: use only the middle/top tokens aligned to the bottom window span.
        top_for_middle = top_arr[t_start : t_start + self._top_cols_for_middle]
        top_for_middle = self._pad(top_for_middle, self._top_cols_for_middle)

        top_for_bottom = top_arr[t_start : t_start + self._top_cols_for_bottom]
        top_for_bottom = self._pad(top_for_bottom, self._top_cols_for_bottom)

        mid_for_bottom = middle_arr[m_start : m_start + self._mid_cols_for_bottom]
        mid_for_bottom = self._pad(mid_for_bottom, self._mid_cols_for_bottom)

        # ── Timing metadata ────────────────────────────────────────────────────
        actual_start_frame = t_start * self.ratios['top']
        start_time_s    = (actual_start_frame * self.hop_length) / self.sample_rate
        total_duration_s = (total_frames * self.hop_length) / self.sample_rate
        fraction         = start_time_s / max(total_duration_s, 1e-6)
        timing = torch.tensor([start_time_s, total_duration_s, fraction], dtype=torch.float32)

        # ── Return (target, cond, second_cond, timing) for the selected level ─
        target    = torch.from_numpy(top_slice).long()
        cond      = None
        second_cond = None

        if self.selected_level == 'top':
            target = torch.from_numpy(top_slice).long()
            # cond and second_cond remain None

        elif self.selected_level == 'middle':
            target = torch.from_numpy(middle_slice).long()
            cond   = torch.from_numpy(top_for_middle).long()  # aligned top slice

        elif self.selected_level == 'bottom':
            target      = torch.from_numpy(bottom_slice).long()
            cond        = torch.from_numpy(mid_for_bottom).long()  # aligned middle slice
            second_cond = torch.from_numpy(top_for_bottom).long()  # aligned top slice

        # Use empty tensors instead of None so DataLoader can collate cleanly.
        # The trainer checks .numel() == 0 or shape to detect "no conditioning".
        cond        = cond        if cond        is not None else torch.empty(0, dtype=torch.long)
        second_cond = second_cond if second_cond is not None else torch.empty(0, dtype=torch.long)

        return target, cond, second_cond, timing

    @staticmethod
    def _shape_2d(arr) -> Tuple[int, int]:
        if torch.is_tensor(arr):
            return int(arr.shape[0]), int(arr.shape[1])
        np_arr = np.asarray(arr)
        return int(np_arr.shape[0]), int(np_arr.shape[1])

    @staticmethod
    def _to_long_tensor(arr) -> torch.Tensor:
        if torch.is_tensor(arr):
            return arr.long()
        return torch.as_tensor(arr, dtype=torch.long)

    @staticmethod
    def _to_float_tensor(arr) -> torch.Tensor:
        if torch.is_tensor(arr):
            return arr.float()
        return torch.as_tensor(arr, dtype=torch.float32)

    @staticmethod
    def _source_stem(file_path: str) -> str:
        return os.path.basename(file_path).replace('.npy', '')

    @staticmethod
    def _pad(arr: np.ndarray, target_t: int) -> np.ndarray:
        """Pad or truncate along axis=0 (time) to exactly target_t rows."""
        if arr.shape[0] < target_t:
            diff = target_t - arr.shape[0]
            arr = np.pad(arr, ((0, diff), (0, 0)), mode='constant')
        return arr[:target_t]


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_dataset_alignment_and_logic():
    print("🚀 Starting JukeboxQuantizedDataset tests...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Mock payload: ratios top=32, mid=16, bot=8
        # 1000 top time-steps × 32 = 32000 raw frames
        total_raw_frames = 32000

        mock_top = np.arange(1000, dtype=np.int64)[:, None].repeat(8, axis=1)   # [1000, 8]
        mock_mid = np.arange(2000, dtype=np.int64)[:, None].repeat(16, axis=1)  # [2000, 16]
        mock_bot = np.arange(4000, dtype=np.int64)[:, None].repeat(32, axis=1)  # [4000, 32]

        quantized_file = os.path.join(tmp_dir, 'test_song_full_quantized.pt')
        torch.save({
            'top': mock_top, 'middle': mock_mid, 'bottom': mock_bot,
            'total_frames': total_raw_frames,
        }, quantized_file)

        level_tfs = {'top': 2048, 'middle': 512, 'bottom': 128}

        # ── Top level ─────────────────────────────────────────────────────────
        ds_top = JukeboxQuantizedDataset(
            tmp_dir, target_time_frames=2048,
            level_target_time_frames=level_tfs, selected_level='top',
        )
        target, cond, second_cond, timing = ds_top[0]
        assert target.shape == (64, 8),   f"Top target shape: {target.shape}"
        assert cond.numel() == 0,         "Top cond should be empty"
        assert second_cond.numel() == 0,  "Top second_cond should be empty"
        print(f"  [Pass] Top level — target={tuple(target.shape)}, cond=empty, second_cond=empty")

        # ── Middle level ──────────────────────────────────────────────────────
        ds_mid = JukeboxQuantizedDataset(
            tmp_dir, target_time_frames=2048,
            level_target_time_frames=level_tfs, selected_level='middle',
        )
        target, cond, second_cond, timing = ds_mid[0]
        # middle window = 512 frames / 16 ratio = 32 time-cols
        # top slice for middle = 32 * (512/2048) = 8 time-cols
        assert target.shape == (32, 16), f"Middle target shape: {target.shape}"
        assert cond.shape[1] == 8,       f"Middle cond freq_bins: {cond.shape}"
        assert second_cond.numel() == 0, "Middle second_cond should be empty"
        # Alignment: cond time-col 0 should match top value at same t_start
        t_val = target[0, 0].item()  # which middle token we started at
        # middle token t_val → top token t_val // 2
        expected_top_start = (t_val * ds_mid.ratios['middle']) // ds_mid.ratios['top']
        assert cond[0, 0].item() == expected_top_start, (
            f"Middle cond misaligned: cond[0,0]={cond[0,0].item()} vs expected={expected_top_start}"
        )
        print(f"  [Pass] Middle level — target={tuple(target.shape)}, cond={tuple(cond.shape)}, aligned ✓")

        # ── Bottom level ──────────────────────────────────────────────────────
        ds_bot = JukeboxQuantizedDataset(
            tmp_dir, target_time_frames=2048,
            level_target_time_frames=level_tfs, selected_level='bottom',
        )
        for i in range(5):
            target, cond, second_cond, timing = ds_bot[0]
            # bottom window = 128 frames / 8 ratio = 16 time-cols
            # mid slice for bottom = 16 * (128/512) = 4 time-cols
            # top slice for bottom = 64 * (128/2048) = 4 time-cols
            assert target.shape[1] == 32,      f"Bottom target freq_bins: {target.shape}"
            assert target.shape[0] == 16,      f"Bottom target time-cols: {target.shape}"
            assert cond.shape[0] == ds_bot._mid_cols_for_bottom
            assert second_cond.shape[0] == ds_bot._top_cols_for_bottom

            # Alignment check using the mock gradient values
            b_val = target[0, 0].item()   # bottom token index
            m_val = cond[0, 0].item()     # middle token index
            t_val = second_cond[0, 0].item()  # top token index

            expected_m = (b_val * ds_bot.ratios['bottom']) // ds_bot.ratios['middle']
            expected_t = (b_val * ds_bot.ratios['bottom']) // ds_bot.ratios['top']
            assert m_val == expected_m, f"Bot→Mid misaligned: m_val={m_val} expected={expected_m}"
            assert t_val == expected_t, f"Bot→Top misaligned: t_val={t_val} expected={expected_t}"

            start_s, total_s, fraction = timing.tolist()
            expected_start_s = (b_val * ds_bot.ratios['bottom'] * ds_bot.hop_length) / ds_bot.sample_rate
            assert abs(start_s - expected_start_s) < 1e-3, f"Timing mismatch: {start_s} vs {expected_start_s}"

            print(f"  [Pass] Bottom crop {i}: B={b_val}, M={m_val}, T={t_val} — all aligned ✓")

        # ── Short song padding ─────────────────────────────────────────────────
        short_file = os.path.join(tmp_dir, 'short_song_full_quantized.pt')
        torch.save({
            'top':    np.zeros((10, 8),  dtype=np.int64),
            'middle': np.zeros((20, 16), dtype=np.int64),
            'bottom': np.zeros((40, 32), dtype=np.int64),
            'total_frames': 320,
        }, short_file)

        ds_short = JukeboxQuantizedDataset(
            tmp_dir,
            file_paths=[short_file.replace('_full_quantized.pt', '.npy')
                        .replace(tmp_dir + '/', '')],  # simulate npy path
            target_time_frames=2048,
            level_target_time_frames=level_tfs,
            selected_level='bottom',
        )
        # Re-init pointing directly at the pt file
        ds_short.files = [short_file]
        ds_short._init_grids()

        target, cond, second_cond, timing = ds_short[0]
        assert target.shape[0] == ds_short._bottom_window_cols, "Short bottom should be padded"
        assert torch.all(target[40:] == 0), "Padded area should be zeros"
        print("  [Pass] Short song padding verified ✓")

        # ── DataLoader collation ───────────────────────────────────────────────
        ds_dl = JukeboxQuantizedDataset(
            tmp_dir, target_time_frames=2048,
            level_target_time_frames=level_tfs, selected_level='bottom',
        )
        loader = DataLoader(ds_dl, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        assert len(batch) == 4, f"Expected 4-tuple, got {len(batch)}"
        print(f"  [Pass] DataLoader batch shapes: target={tuple(batch[0].shape)}, "
              f"cond={tuple(batch[1].shape)}, second_cond={tuple(batch[2].shape)}, timing={tuple(batch[3].shape)}")

    print("\nAll tests passed successfully!")


if __name__ == '__main__':
    test_dataset_alignment_and_logic()
