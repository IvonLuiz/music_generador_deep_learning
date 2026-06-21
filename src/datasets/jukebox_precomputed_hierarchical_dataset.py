import json
import os
import sys
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

# Add 'src' to sys.path so the script can be run directly from the project root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metadata.title_key import (
    UNKNOWN_KEY_ID,
    build_title_key_metadata_by_source,
    key_metadata_for_path,
    unknown_key_metadata,
)


class JukeboxQuantizedDataset(Dataset):
    """
    Dataset for hierarchical transformer prior training.

    Returns (target, cond, second_cond, timing, metadata) for every level:
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
        window_parity: str = 'all',
        metadata_path: Optional[str] = None,
        key_infer_missing_mode_as: str = 'major',
        key_dropout_prob: float = 0.0,
        timing_dropout_prob: float = 0.0,
    ):
        """
        Args:
            quantized_path:           Directory containing windowed quantized files and windowed_manifest.jsonl.
            file_paths:               Optional list of source paths used to filter manifest source stems.
            target_time_frames:       Top-level window size in raw spectrogram frames.
            level_target_time_frames: Dict with per-level window sizes, e.g.
                                      {'top': 2048, 'middle': 512, 'bottom': 128}.
            selected_level:           Which prior is being trained ('top'/'middle'/'bottom').
            sample_rate:              Audio sample rate (used for timing metadata).
            hop_length:               STFT hop length (used for timing metadata).
            window_parity:            Window subset for windowed datasets: all/even/odd.
        """
        self.selected_level = selected_level
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.quantized_path = quantized_path
        self.mode = 'windowed'
        self.window_entries = []
        self.window_parity = self._normalize_window_parity(window_parity)
        self.key_unknown_id = UNKNOWN_KEY_ID
        self.key_dropout_prob = float(key_dropout_prob)
        self.timing_dropout_prob = float(timing_dropout_prob)
        self.title_key_metadata_by_source = build_title_key_metadata_by_source(
            metadata_path,
            infer_missing_mode_as=key_infer_missing_mode_as,
        )
        self.sidecar_metadata_by_source = self._load_sidecar_metadata()

        lvl = level_target_time_frames or {}
        self.top_tf    = int(lvl.get('top',    target_time_frames))
        self.middle_tf = int(lvl.get('middle', target_time_frames))
        self.bottom_tf = int(lvl.get('bottom', target_time_frames))

        if not self._init_windowed_files(file_paths=file_paths):
            raise FileNotFoundError(
                f"No windowed quantized entries found in {quantized_path}. "
                "Expected windowed_manifest.jsonl produced by preprocess_quantization.py."
            )
        self.set_window_parity(self.window_parity)

        # Peek at the first file to learn the actual token grid shapes.
        # This replaces all hardcoded ratio constants.
        self._init_grids()

    def _load_sidecar_metadata(self) -> Dict[str, dict]:
        sidecar_path = os.path.join(self.quantized_path, 'source_metadata.json')
        if not os.path.isfile(sidecar_path):
            return {}
        with open(sidecar_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}

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

        entries.sort(key=lambda record: (
            record.get('source_stem', ''),
            int(record.get('start_frame', 0)),
            record.get('file', ''),
        ))
        current_source = None
        window_index = -1
        for record in entries:
            source_stem = record.get('source_stem', '')
            if source_stem != current_source:
                current_source = source_stem
                window_index = 0
            else:
                window_index += 1
            record['window_index_in_song'] = int(window_index)
            record['window_parity'] = 'even' if window_index % 2 == 0 else 'odd'

        self.mode = 'windowed'
        self.window_entries = entries
        self.files = [os.path.join(self.quantized_path, entry['file']) for entry in entries]
        return True

    @staticmethod
    def _normalize_window_parity(window_parity: str) -> str:
        parity = str(window_parity or 'all').strip().lower()
        aliases = {
            'none': 'all',
            'full': 'all',
            'both': 'all',
            'non_overlapping': 'even',
            'non-overlapping': 'even',
            'fixed': 'even',
            'first': 'even',
        }
        parity = aliases.get(parity, parity)
        if parity not in ('all', 'even', 'odd'):
            raise ValueError("window_parity must be one of: all, even, odd")
        return parity

    def set_window_parity(self, window_parity: str) -> None:
        """
        Keep only one overlap parity active for windowed datasets.

        The parity is based on each window's order within its source song after
        sorting by start frame. With 50% overlap, even windows are effectively
        the non-overlapping 0, 2, 4... anchors; odd windows are the shifted view.
        """
        self.window_parity = self._normalize_window_parity(window_parity)

        if self.window_parity == 'all':
            active_entries = list(self.window_entries)
        else:
            active_entries = [
                entry for entry in self.window_entries
                if entry.get('window_parity') == self.window_parity
            ]
        if not active_entries:
            raise ValueError(
                f"No windowed quantized entries remain after applying window_parity={self.window_parity!r}."
            )
        self.files = [os.path.join(self.quantized_path, entry['file']) for entry in active_entries]
        self.active_window_entries = active_entries

    def indices_for_window_parity(self, window_parity: str) -> List[int]:
        """Return dataset indices for a fixed overlap parity without mutating the dataset."""
        parity = self._normalize_window_parity(window_parity)
        if parity == 'all':
            return list(range(len(self.files)))
        return [
            idx for idx, entry in enumerate(self.window_entries)
            if entry.get('window_parity') == parity
        ]

    def indices_for_random_window_parity_per_song(self, seed: int) -> List[int]:
        """Choose even or odd windows independently for each song."""
        rng = np.random.default_rng(seed)
        available_parities = {}
        for entry in self.window_entries:
            source_stem = entry.get('source_stem', '')
            available_parities.setdefault(source_stem, set()).add(entry.get('window_parity', 'even'))
        chosen = {
            source_stem: sorted(parities)[int(rng.integers(0, len(parities)))]
            for source_stem, parities in sorted(available_parities.items())
        }
        return [
            idx for idx, entry in enumerate(self.window_entries)
            if entry.get('window_parity') == chosen.get(entry.get('source_stem', ''), 'even')
        ]

    def window_parity_counts(self) -> Dict[str, int]:
        even = sum(1 for entry in self.window_entries if entry.get('window_parity') == 'even')
        odd = sum(1 for entry in self.window_entries if entry.get('window_parity') == 'odd')
        return {'all': len(self.window_entries), 'even': even, 'odd': odd}

    def _init_grids(self):
        """Read one payload to discover fixed-window token grids."""
        if not self.files:
            raise ValueError("No quantized .pt files found.")
        payload = torch.load(self.files[0], weights_only=False)
        if payload.get('format') != 'windowed_v1':
            raise ValueError(
                f"Unsupported quantized payload format {payload.get('format')!r}. "
                "JukeboxQuantizedDataset now requires windowed_v1 payloads."
            )
        self._init_windowed_grids(payload)

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

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self._getitem_windowed(idx)

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

        metadata = self._sample_metadata(payload, self.files[idx % len(self.files)])
        return target, cond, second_cond, timing, metadata

    def _sample_metadata(self, payload: dict, file_path: str) -> dict:
        metadata = self._resolve_key_metadata(payload, file_path)
        key_id = int(metadata.get('key_id', self.key_unknown_id))
        if self.key_dropout_prob > 0.0 and np.random.random() < self.key_dropout_prob:
            key_id = self.key_unknown_id
            metadata = {
                **metadata,
                'key_id': key_id,
                'key_source': 'dropout',
            }
        timing_mask = True
        if self.timing_dropout_prob > 0.0 and np.random.random() < self.timing_dropout_prob:
            timing_mask = False
        return {
            'key_id': torch.tensor(key_id, dtype=torch.long),
            'timing_mask': torch.tensor(timing_mask, dtype=torch.bool),
            'key_label': metadata.get('key_label', 'unknown'),
            'key_source': metadata.get('key_source', 'unknown'),
        }

    def _resolve_key_metadata(self, payload: dict, file_path: str) -> dict:
        payload_metadata = payload.get('metadata')
        if isinstance(payload_metadata, dict) and 'key_id' in payload_metadata:
            return dict(payload_metadata)

        for source_key in self._metadata_lookup_keys(payload, file_path):
            if source_key in self.sidecar_metadata_by_source:
                value = self.sidecar_metadata_by_source[source_key]
                if isinstance(value, dict) and 'key_id' in value:
                    return dict(value)

        title_metadata = key_metadata_for_path(file_path, self.title_key_metadata_by_source)
        if int(title_metadata.get('key_id', self.key_unknown_id)) != self.key_unknown_id:
            return title_metadata
        for source_key in self._metadata_lookup_keys(payload, file_path):
            title_metadata = key_metadata_for_path(source_key, self.title_key_metadata_by_source)
            if int(title_metadata.get('key_id', self.key_unknown_id)) != self.key_unknown_id:
                return title_metadata

        return unknown_key_metadata()

    @staticmethod
    def _metadata_lookup_keys(payload: dict, file_path: str) -> List[str]:
        keys = []
        for key in ('source_stem', 'source_basename', 'source_path'):
            value = payload.get(key)
            if value:
                keys.append(str(value))
        basename = os.path.basename(file_path)
        stem = os.path.splitext(basename)[0]
        keys.extend([
            basename,
            stem,
            stem.split('__start_')[0],
        ])
        deduped = []
        seen = set()
        for key in keys:
            if key not in seen:
                seen.add(key)
                deduped.append(key)
        return deduped

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
        return os.path.splitext(os.path.basename(file_path))[0]
