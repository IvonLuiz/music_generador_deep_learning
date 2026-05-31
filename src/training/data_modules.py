from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset

from datasets.quantized_dataset import PixelCNNQuantizedDataset, TwoLevelPixelCNNQuantizedDataset
from datasets.raw_audio_dataset import RawAudioWindowDataset, collate_audio_windows, list_audio_files
from datasets.spectrogram_dataset import MmapSpectrogramDataset
from processing.gpu_audio_augmentation import GPUAudioToMelSpectrogram
from train_scripts.jukebox_utils import split_paths_by_maestro_metadata, split_train_val_paths
from utils import load_maestro

from .common import DataBundle, dataloader_kwargs, estimate_preprocessed_variance


def _select_global_train_parity(mode: str, epoch: int, seed: int) -> str:
    mode = str(mode or 'alternate').strip().lower()
    if mode == 'all':
        return 'all'
    if mode == 'alternate':
        return 'even' if epoch % 2 == 0 else 'odd'
    if mode == 'random_per_song':
        return 'random_per_song'
    raise ValueError(f'Unsupported train window parity mode: {mode}')


def _indices_for_train_epoch(dataset, mode: str, epoch: int, seed: int):
    parity = _select_global_train_parity(mode, epoch, seed)
    if parity == 'random_per_song' and hasattr(dataset, 'indices_for_random_window_parity_per_song'):
        return dataset.indices_for_random_window_parity_per_song(seed + epoch), parity
    if hasattr(dataset, 'indices_for_window_parity'):
        return dataset.indices_for_window_parity(parity), parity
    return list(range(len(dataset))), 'all'


class WindowParityEpochSampler(Sampler):
    """Shuffle a fixed overlap-parity subset, changing that subset each epoch."""

    def __init__(self, dataset, mode: str, seed: int):
        self.dataset = dataset
        self.mode = str(mode or 'alternate').strip().lower()
        self.seed = int(seed)
        self.epoch = 0
        self.current_parity = 'all'
        self.indices = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        indices, parity = _indices_for_train_epoch(self.dataset, self.mode, self.epoch, self.seed)
        rng = np.random.default_rng(self.seed + self.epoch)
        self.indices = list(indices)
        rng.shuffle(self.indices)
        self.current_parity = parity

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def _load_min_max_values(dataset_cfg: dict):
    path = dataset_cfg.get('min_max_values_path')
    if not path:
        return None
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        path = os.path.join(path, 'min_max_values.pkl')
    if not os.path.isfile(path):
        print(f'Warning: min_max_values_path not found: {path}')
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def _split_audio_paths(all_file_paths, dataset_cfg, validation_split, seed):
    metadata_path = dataset_cfg.get('metadata_path')
    if metadata_path and os.path.isfile(os.path.expanduser(metadata_path)):
        return split_paths_by_maestro_metadata(all_file_paths, dataset_cfg)
    train_paths, val_paths = split_train_val_paths(
        all_file_paths,
        dataset_cfg,
        validation_split=validation_split,
        seed=seed,
    )
    return train_paths, val_paths or [], []


class SpectrogramWindowDataModule:
    def __init__(self, config: dict):
        self.config = config

    def setup(self, device: torch.device) -> DataBundle:
        dataset_cfg = self.config['dataset']
        training_cfg = self.config['training']
        spectrograms_path = dataset_cfg.get('processed_path')
        if not spectrograms_path:
            raise ValueError('dataset.processed_path is required for image/spectrogram input mode.')

        target_time_frames = int(dataset_cfg.get('target_time_frames', 256))
        x_all, file_paths = load_maestro(spectrograms_path, target_time_frames, debug_print=False)
        if len(x_all) == 0:
            raise ValueError(f'No spectrograms were loaded from {spectrograms_path}.')
        data_variance = float(np.var(x_all))
        validation_split = float(training_cfg.get('validation_split', 0.0))

        if validation_split > 0:
            num_samples = len(x_all)
            num_val = int(num_samples * validation_split)
            num_train = num_samples - num_val
            indices = np.random.permutation(num_samples)
            train_indices = indices[:num_train]
            val_indices = indices[num_train:]
            train_dataset = MmapSpectrogramDataset(x_all, train_indices)
            val_dataset = MmapSpectrogramDataset(x_all, val_indices)
            file_paths_np = np.asarray(file_paths)
            train_file_paths = file_paths_np[train_indices].tolist()
            val_file_paths = file_paths_np[val_indices].tolist()
        else:
            train_dataset = MmapSpectrogramDataset(x_all)
            val_dataset = None
            train_file_paths = list(file_paths)
            val_file_paths = []

        kwargs = dataloader_kwargs(training_cfg)
        batch_size = int(training_cfg['batch_size'])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        val_loader = (
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
            if val_dataset is not None else None
        )
        print(
            f'Spectrogram dataset: train={len(train_dataset)}, '
            f'val={len(val_dataset) if val_dataset is not None else 0}, '
            f'target_time_frames={target_time_frames}, variance={data_variance:.6f}'
        )
        return DataBundle(
            train_loader=train_loader,
            val_loader=val_loader,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            data_variance=data_variance,
            min_max_values=_load_min_max_values(dataset_cfg),
            train_file_paths=train_file_paths,
            val_file_paths=val_file_paths,
            metadata={'input_mode': 'image'},
        )


class AudioWindowDataModule:
    def __init__(self, config: dict, input_mode_override: Optional[str] = None):
        self.config = config
        self.input_mode_override = input_mode_override

    def setup(self, device: torch.device) -> DataBundle:
        dataset_cfg = self.config['dataset']
        training_cfg = self.config['training']
        raw_audio_path = dataset_cfg.get('raw_path')
        if not raw_audio_path:
            raise ValueError('dataset.raw_path is required for audio input mode.')

        audio_cfg = dataset_cfg.get('audio', {})
        sample_rate = int(dataset_cfg.get('sample_rate', 22050))
        hop_length = int(dataset_cfg.get('hop_length', 256))
        frame_size = int(dataset_cfg.get('frame_size', 2048))
        target_time_frames = int(dataset_cfg.get('target_time_frames', 256))
        n_mels = int(dataset_cfg.get('n_mels', 256))
        validation_split = float(training_cfg.get('validation_split', 0.0))
        seed = int(training_cfg.get('seed', 42))

        print(f'Scanning for raw audio files in {raw_audio_path}...')
        all_file_paths = list_audio_files(raw_audio_path, extensions=audio_cfg.get('extensions'))
        if not all_file_paths:
            raise FileNotFoundError(f'No audio files found in {raw_audio_path}')

        train_file_paths, val_file_paths, test_file_paths = _split_audio_paths(
            all_file_paths,
            dataset_cfg,
            validation_split=validation_split,
            seed=seed,
        )
        splits_cfg = dataset_cfg.get('splits', {})
        if bool(splits_cfg.get('include_test_in_train', False)):
            train_file_paths = sorted(train_file_paths + test_file_paths)
        if not train_file_paths:
            raise ValueError('The audio split produced no training files.')

        examples_per_file = int(audio_cfg.get('examples_per_file', 1))
        downmix_cfg = audio_cfg.get('downmix', {})
        pitch_cfg = audio_cfg.get('pitch_shift', {})
        pitch_enabled = bool(pitch_cfg.get('enabled', True))
        pitch_choices = pitch_cfg.get('semitone_choices')
        pitch_range = pitch_cfg.get('semitone_range', [-2.0, 2.0])
        if pitch_choices:
            pitch_range = [min(pitch_choices), max(pitch_choices)]
        if not pitch_enabled:
            pitch_range = [0.0, 0.0]

        target_num_samples = max(1, (target_time_frames - 1) * hop_length)
        train_dataset = RawAudioWindowDataset(
            train_file_paths,
            target_sample_rate=sample_rate,
            target_num_samples=target_num_samples,
            min_pitch_shift_semitones=float(pitch_range[0]),
            max_pitch_shift_semitones=float(pitch_range[1]),
            examples_per_file=examples_per_file,
            crop_strategy='random',
        )
        val_dataset = None
        if val_file_paths:
            val_dataset = RawAudioWindowDataset(
                val_file_paths,
                target_sample_rate=sample_rate,
                target_num_samples=target_num_samples,
                min_pitch_shift_semitones=0.0,
                max_pitch_shift_semitones=0.0,
                examples_per_file=1,
                crop_strategy='non_overlapping',
            )

        downmix_weight_range = downmix_cfg.get('weight_range', [0.0, 1.0])
        batch_preprocessor = GPUAudioToMelSpectrogram(
            sample_rate=sample_rate,
            target_time_frames=target_time_frames,
            n_fft=frame_size,
            hop_length=hop_length,
            n_mels=n_mels,
            random_downmix=bool(downmix_cfg.get('enabled', True)),
            downmix_weight_min=float(downmix_weight_range[0]),
            downmix_weight_max=float(downmix_weight_range[1]),
            pitch_shift_enabled=pitch_enabled,
            min_pitch_shift_semitones=float(pitch_range[0]),
            max_pitch_shift_semitones=float(pitch_range[1]),
            pitch_shift_choices=pitch_choices,
            resample_lowpass_filter_width=int(audio_cfg.get('resample_lowpass_filter_width', 64)),
            resample_chunk_size=int(audio_cfg.get('resample_chunk_size', 8192)),
            max_torchaudio_resample_factor=int(audio_cfg.get('max_torchaudio_resample_factor', 256)),
        ).to(device)

        kwargs = dataloader_kwargs(training_cfg)
        batch_size = int(training_cfg['batch_size'])
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_audio_windows,
            **kwargs,
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_audio_windows,
                **kwargs,
            )
            if val_dataset is not None else None
        )
        data_variance = estimate_preprocessed_variance(
            train_loader,
            batch_preprocessor,
            device,
            max_samples=int(training_cfg.get('data_variance_samples', 1000)),
        )
        print(
            'Raw audio augmentation: '
            f'train_files={len(train_file_paths)}, val_files={len(val_file_paths)}, '
            f'test_files={len(test_file_paths)}, train_windows={len(train_dataset)}, '
            f'val_windows={len(val_dataset) if val_dataset is not None else 0}, '
            f'examples_per_file={examples_per_file}, target_num_samples={target_num_samples}, '
            f'downmix={bool(downmix_cfg.get("enabled", True))}, pitch_enabled={pitch_enabled}, '
            f'pitch_choices={pitch_choices if pitch_choices else "continuous"}, pitch_range={pitch_range}'
        )
        return DataBundle(
            train_loader=train_loader,
            val_loader=val_loader,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            data_variance=data_variance,
            batch_preprocessor=batch_preprocessor,
            collate_fn=collate_audio_windows,
            min_max_values={},
            train_file_paths=train_file_paths,
            val_file_paths=val_file_paths,
            metadata={'input_mode': 'audio'},
        )


class QuantizedPriorDataModule:
    def __init__(
        self,
        config: dict,
        variant: str = 'single',
    ):
        self.config = config
        self.variant = variant

    def setup(self, device: torch.device) -> DataBundle:
        dataset_cfg = self.config['dataset']
        input_mode = str(
            dataset_cfg.get('input_mode', 'quantized' if dataset_cfg.get('quantized_path') else 'spectrogram')
        ).strip().lower()
        if input_mode != 'quantized':
            raise ValueError(
                "PixelCNN training now requires dataset.input_mode='quantized'. "
                'Run src/processing/preprocess_vqvae_quantization.py first and set dataset.quantized_path.'
            )
        if self.variant == 'single':
            return self._setup_precomputed(PixelCNNQuantizedDataset)
        if self.variant == 'two_level':
            return self._setup_precomputed(TwoLevelPixelCNNQuantizedDataset)
        raise ValueError(f'Unsupported quantized prior variant: {self.variant}')

    def _setup_precomputed(self, dataset_cls) -> DataBundle:
        dataset_cfg = self.config['dataset']
        training_cfg = self.config['training']
        quantized_path = dataset_cfg.get('quantized_path')
        if not quantized_path:
            raise ValueError("dataset.quantized_path is required when dataset.input_mode='quantized'.")

        splits_cfg = dataset_cfg.get('splits', {})
        train_splits = dataset_cfg.get('train_splits', splits_cfg.get('train', ['train']))
        validation_splits = dataset_cfg.get(
            'validation_splits',
            splits_cfg.get('validation', [dataset_cfg.get('validation_split_name', 'validation')]),
        )
        if bool(splits_cfg.get('include_test_in_train', False)) and 'test' not in train_splits:
            train_splits = list(train_splits) + ['test']

        manifest_file = dataset_cfg.get('manifest_file', 'pixelcnn_quantized_manifest.jsonl')
        preload = bool(dataset_cfg.get('preload', False))
        train_dataset = dataset_cls(
            quantized_path=quantized_path,
            split=train_splits,
            manifest_file=manifest_file,
            preload=preload,
        )
        val_dataset = dataset_cls(
            quantized_path=quantized_path,
            split=validation_splits,
            manifest_file=manifest_file,
            preload=preload,
        )

        parity_cfg = dataset_cfg.get('window_parity', {})
        train_window_parity_mode = dataset_cfg.get('train_window_parity_mode', parity_cfg.get('train_mode', 'alternate'))
        validation_window_parity = dataset_cfg.get('validation_window_parity', parity_cfg.get('validation', 'even'))
        train_sampler = WindowParityEpochSampler(
            train_dataset,
            mode=train_window_parity_mode,
            seed=int(training_cfg.get('seed', 42)),
        )
        if hasattr(val_dataset, 'indices_for_window_parity'):
            val_indices = val_dataset.indices_for_window_parity(validation_window_parity)
        else:
            val_indices = list(range(len(val_dataset)))
        val_subset = Subset(val_dataset, val_indices)

        kwargs = dataloader_kwargs(training_cfg)
        batch_size = int(training_cfg['batch_size'])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, **kwargs)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, **kwargs)
        if hasattr(train_dataset, 'window_parity_counts'):
            counts = train_dataset.window_parity_counts()
            print(
                f'Training window parity mode: {train_window_parity_mode} '
                f"(all={counts['all']}, even={counts['even']}, odd={counts['odd']})"
            )
        print(
            f'Validation window parity: {validation_window_parity} '
            f'(active={len(val_subset)}, full={len(val_dataset)})'
        )
        return DataBundle(
            train_loader=train_loader,
            val_loader=val_loader,
            train_dataset=train_dataset,
            val_dataset=val_subset,
            num_embeddings=train_dataset.num_embeddings,
            input_size=train_dataset.input_size,
            metadata={'input_mode': 'quantized', 'variant': self.variant},
        )


def build_vqvae_data_module(config: dict, input_mode_override: Optional[str] = None):
    dataset_cfg = config['dataset']
    input_mode = str(input_mode_override or dataset_cfg.get('input_mode', 'spectrogram')).strip().lower()
    if input_mode in ('spectrogram', 'spec', 'npy'):
        input_mode = 'image'
    dataset_cfg['input_mode'] = input_mode
    if input_mode == 'audio':
        return AudioWindowDataModule(config, input_mode_override=input_mode)
    if input_mode == 'image':
        return SpectrogramWindowDataModule(config)
    raise ValueError("dataset.input_mode must be either 'audio' or 'image' (alias: 'spectrogram').")
