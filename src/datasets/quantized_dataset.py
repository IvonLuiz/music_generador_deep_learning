import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


PIXELCNN_MANIFEST = 'pixelcnn_quantized_manifest.jsonl'
PIXELCNN_CONFIG = 'pixelcnn_quantization_config.json'


class PixelCNNQuantizedDataset(Dataset):
    """
    Lazy dataset for single VQ-VAE PixelCNN training from precomputed index files.

    Expected layout is produced by processing/preprocess_vqvae_quantization.py:
      - pixelcnn_quantized_manifest.jsonl
      - pixelcnn_quantization_config.json
      - *_vqvae_indices.pt files containing an "indices" tensor
    """

    def __init__(
        self,
        quantized_path: str,
        split='train',
        manifest_file: str = PIXELCNN_MANIFEST,
        preload: bool = False,
    ):
        self.quantized_path = os.path.abspath(os.path.expanduser(quantized_path))
        self.splits = self._normalize_splits(split)
        self.split = '+'.join(self.splits)
        self.manifest_file = manifest_file
        self.preload = bool(preload)
        self.config = self._load_config()
        self.entries = self._load_manifest()
        if not self.entries:
            raise ValueError(
                f"No quantized PixelCNN entries found for split={self.split!r} "
                f"in {os.path.join(self.quantized_path, self.manifest_file)}"
            )

        self.files = [os.path.join(self.quantized_path, entry['file']) for entry in self.entries]
        missing = [path for path in self.files if not os.path.isfile(path)]
        if missing:
            preview = '\n'.join(f'  - {path}' for path in missing[:10])
            extra = '' if len(missing) <= 10 else f'\n  ... and {len(missing) - 10} more'
            raise FileNotFoundError(f"Missing quantized index files:\n{preview}{extra}")

        first = self._load_payload(0)
        first_indices = self._extract_indices(first, self.files[0])
        self.index_shape = tuple(int(x) for x in first_indices.shape)
        self.num_embeddings = self._resolve_num_embeddings()

        self._cache = None
        if self.preload:
            self._cache = [self._load_indices(i) for i in tqdm(range(len(self.files)), desc=f'Preloading PixelCNN indices [{self.split}]')]

        print(
            f"[PixelCNNQuantizedDataset] split={self.split}, files={len(self.files)}, "
            f"index_shape={self.index_shape}, num_embeddings={self.num_embeddings}"
        )

    @classmethod
    def _normalize_splits(cls, split) -> list:
        if isinstance(split, (list, tuple, set)):
            values = list(split)
        else:
            values = str(split or 'train').replace(',', ' ').split()
        splits = [cls._normalize_split(value) for value in values if str(value).strip()]
        return splits or ['train']

    @staticmethod
    def _normalize_split(split: str) -> str:
        split = str(split or 'train').strip().lower()
        aliases = {
            'val': 'validation',
            'valid': 'validation',
            'dev': 'validation',
            'all': 'all',
            '*': 'all',
        }
        return aliases.get(split, split)

    def _load_config(self):
        config_path = os.path.join(self.quantized_path, PIXELCNN_CONFIG)
        if not os.path.isfile(config_path):
            return {}
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}

    def _load_manifest(self):
        manifest_path = os.path.join(self.quantized_path, self.manifest_file)
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Quantized manifest not found: {manifest_path}")

        entries = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record_split = self._normalize_split(record.get('split', 'unknown'))
                if 'all' not in self.splits and record_split not in self.splits:
                    continue
                entries.append(record)

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
        return entries

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

    def indices_for_window_parity(self, window_parity: str):
        parity = self._normalize_window_parity(window_parity)
        if parity == 'all':
            return list(range(len(self.entries)))
        return [
            idx for idx, entry in enumerate(self.entries)
            if entry.get('window_parity') == parity
        ]

    def indices_for_random_window_parity_per_song(self, seed: int):
        rng = np.random.default_rng(seed)
        available_parities = {}
        for entry in self.entries:
            source_stem = entry.get('source_stem', '')
            available_parities.setdefault(source_stem, set()).add(entry.get('window_parity', 'even'))
        chosen = {
            source_stem: sorted(parities)[int(rng.integers(0, len(parities)))]
            for source_stem, parities in sorted(available_parities.items())
        }
        return [
            idx for idx, entry in enumerate(self.entries)
            if entry.get('window_parity') == chosen.get(entry.get('source_stem', ''), 'even')
        ]

    def window_parity_counts(self):
        even = sum(1 for entry in self.entries if entry.get('window_parity') == 'even')
        odd = sum(1 for entry in self.entries if entry.get('window_parity') == 'odd')
        return {'all': len(self.entries), 'even': even, 'odd': odd}

    def _resolve_num_embeddings(self):
        value = self.config.get('num_embeddings')
        if value is None:
            return None
        return int(value)

    def _load_payload(self, idx: int):
        return torch.load(self.files[idx], map_location='cpu', weights_only=False)

    @staticmethod
    def _extract_indices(payload, file_path: str):
        if not isinstance(payload, dict) or 'indices' not in payload:
            raise KeyError(f"Quantized payload at {file_path} does not contain an 'indices' tensor.")
        indices = payload['indices']
        if not torch.is_tensor(indices):
            indices = torch.as_tensor(indices)
        if indices.ndim != 2:
            raise ValueError(f"Expected 2D index grid in {file_path}, got shape {tuple(indices.shape)}")
        return indices.long()

    def _load_indices(self, idx: int):
        return self._extract_indices(self._load_payload(idx), self.files[idx])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self._cache is not None:
            return self._cache[idx]
        return self._load_indices(idx)
