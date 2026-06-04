from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset

from datasets.jukebox_precomputed_hierarchical_dataset import JukeboxQuantizedDataset
from datasets.quantized_dataset import PixelCNNQuantizedDataset, TwoLevelPixelCNNQuantizedDataset
from datasets.raw_audio_dataset import RawAudioWindowDataset, collate_audio_windows, list_audio_files
from processing.gpu_audio_augmentation import GPUAudioToMelSpectrogram
from train_scripts.jukebox_utils import split_paths_by_maestro_metadata, split_train_val_paths
from utils import list_npy_files

from .common import DataBundle, dataloader_kwargs, estimate_preprocessed_variance


def _select_global_train_parity(mode: str, epoch: int, seed: int) -> str:
    """!
    @brief Resolve which overlap-parity subset should be active this epoch.
    """
    mode = str(mode or 'alternate').strip().lower()
    if mode == 'all':
        return 'all'
    if mode == 'alternate':
        return 'even' if epoch % 2 == 0 else 'odd'
    if mode == 'random_per_song':
        return 'random_per_song'
    raise ValueError(f'Unsupported train window parity mode: {mode}')


def _indices_for_train_epoch(dataset, mode: str, epoch: int, seed: int):
    """!
    @brief Return dataset indices for the selected train parity strategy.
    """
    parity = _select_global_train_parity(mode, epoch, seed)
    if parity == 'random_per_song' and hasattr(dataset, 'indices_for_random_window_parity_per_song'):
        return dataset.indices_for_random_window_parity_per_song(seed + epoch), parity
    if hasattr(dataset, 'indices_for_window_parity'):
        return dataset.indices_for_window_parity(parity), parity
    return list(range(len(dataset))), 'all'


class WindowParityEpochSampler(Sampler):
    """!
    @brief Shuffle one overlap-parity subset and change it when set_epoch is called.

    The PixelCNN quantized datasets can contain overlapping even and odd windows.
    Alternating them by epoch keeps each epoch smaller while still exposing both
    views of the song over training.
    """

    def __init__(self, dataset, mode: str, seed: int):
        self.dataset = dataset
        self.mode = str(mode or 'alternate').strip().lower()
        self.seed = int(seed)
        self.epoch = 0
        self.current_parity = 'all'
        self.indices = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        """!
        @brief Select and shuffle the active indices for an epoch.
        """
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


def _split_audio_paths(all_file_paths, dataset_cfg, validation_split, seed):
    """!
    @brief Split raw audio paths using Maestro metadata when available.
    """
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


class AudioWindowDataModule:
    """!
    @brief Data module for raw-audio VQ-VAE training with GPU Mel preprocessing.

    The dataset returns waveform windows. GPUAudioToMelSpectrogram converts those
    windows to normalized Mel spectrograms inside TrainingAdapter.prepare_batch,
    allowing pitch/downmix augmentation to happen online.
    """

    def __init__(self, config: dict):
        self.config = config

    def setup(self, device: torch.device) -> DataBundle:
        """!
        @brief Build raw-audio datasets, loaders and the GPU batch preprocessor.
        """
        dataset_cfg = self.config['dataset']
        training_cfg = self.config['training']
        input_mode = str(dataset_cfg.get('input_mode', 'audio')).strip().lower()
        if input_mode != 'audio':
            raise ValueError("VQ-VAE training now requires dataset.input_mode='audio'.")
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
            # Prior-like generation tasks often reserve validation for monitoring
            # and merge test into train to maximize musical coverage.
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
            metadata={'input_mode': 'audio'},
        )


class QuantizedPriorDataModule:
    """!
    @brief Data module for PixelCNN priors trained on precomputed VQ-VAE indices.

    Supports both single-level indices and two-level hierarchical indices. It
    mirrors the Jukebox parity workflow by alternating even and odd overlapping
    windows for training while keeping validation fixed.
    """

    def __init__(
        self,
        config: dict,
        variant: str = 'single',
    ):
        self.config = config
        self.variant = variant

    def setup(self, device: torch.device) -> DataBundle:
        """!
        @brief Select the appropriate quantized dataset variant and create loaders.
        """
        dataset_cfg = self.config['dataset']
        input_mode = str(dataset_cfg.get('input_mode', 'quantized')).strip().lower()
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
        """!
        @brief Build train/validation loaders from a quantized manifest.
        """
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
            # Keep validation deterministic so early stopping compares epochs
            # against the same window subset.
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


class JukeboxVQVAEDataModule:
    """!
    @brief Data module for one selected Jukebox VQ-VAE level.
    """

    def __init__(
        self,
        config: dict,
        level: str,
    ):
        self.config = config
        self.level = str(level).strip().lower()

    def setup(self, device: torch.device) -> DataBundle:
        """!
        @brief Build level-specific datasets for Jukebox VQ-VAE training.
        """
        dataset_cfg = self.config['dataset']
        model_cfg = self.config['model']
        training_cfg = self.config['training']
        level_profile = model_cfg.get('level_profiles', {}).get(self.level)
        if not level_profile:
            raise ValueError(f"Missing model.level_profiles.{self.level} in Jukebox config.")

        target_time_frames = int(level_profile.get('target_time_frames', dataset_cfg.get('target_time_frames', 2048)))
        dataset_cfg['target_time_frames'] = target_time_frames
        input_mode = str(dataset_cfg.get('input_mode', 'audio')).strip().lower()
        if input_mode != 'audio':
            raise ValueError("Jukebox VQ-VAE training now requires dataset.input_mode='audio'.")

        sample_rate = int(dataset_cfg.get('sample_rate', 22050))
        hop_length = int(dataset_cfg.get('hop_length', 256))
        frame_size = int(dataset_cfg.get('frame_size', 2048))
        n_mels = int(dataset_cfg.get('n_mels', 256))
        data_variance_samples = int(training_cfg.get('data_variance_samples', 1000))

        train_dataset, val_dataset, train_file_paths, val_file_paths, batch_preprocessor = self._setup_audio(
            dataset_cfg,
            target_time_frames,
            sample_rate,
            hop_length,
            frame_size,
            n_mels,
            device,
        )
        collate_fn = collate_audio_windows
        data_variance = None

        kwargs = dataloader_kwargs(training_cfg)
        batch_size = int(training_cfg['batch_size'])
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            **kwargs,
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                **kwargs,
            )
            if val_dataset is not None else None
        )

        if data_variance is None:
            data_variance = estimate_preprocessed_variance(
                train_loader,
                batch_preprocessor,
                device,
                max_samples=data_variance_samples,
            )

        print(
            f'Jukebox VQ-VAE {self.level}: input_mode={input_mode}, '
            f'train={len(train_dataset)}, val={len(val_dataset) if val_dataset is not None else 0}, '
            f'target_time_frames={target_time_frames}, data_variance={float(data_variance):.6f}'
        )
        return DataBundle(
            train_loader=train_loader,
            val_loader=val_loader,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            data_variance=float(data_variance),
            batch_preprocessor=batch_preprocessor,
            collate_fn=collate_fn,
            metadata={
                'family': 'jukebox_vqvae',
                'level': self.level,
                'input_mode': input_mode,
                'target_time_frames': target_time_frames,
                'sample_rate': sample_rate,
                'hop_length': hop_length,
                'frame_size': frame_size,
                'n_mels': n_mels,
            },
        )

    def _setup_audio(
        self,
        dataset_cfg,
        target_time_frames,
        sample_rate,
        hop_length,
        frame_size,
        n_mels,
        device,
    ):
        raw_audio_path = dataset_cfg.get('raw_path')
        if not raw_audio_path:
            raise ValueError('dataset.raw_path is required for Jukebox audio input mode.')
        audio_cfg = dataset_cfg.get('audio', {})
        print(f'Scanning for raw audio files in {raw_audio_path}...')
        all_file_paths = list_audio_files(raw_audio_path, extensions=audio_cfg.get('extensions'))
        if not all_file_paths:
            raise FileNotFoundError(f'No audio files found in {raw_audio_path}')
        train_file_paths, val_file_paths, _test_file_paths = split_paths_by_maestro_metadata(all_file_paths, dataset_cfg)

        examples_by_level = audio_cfg.get('examples_per_file_by_level', {})
        examples_per_file = int(examples_by_level.get(self.level, audio_cfg.get('examples_per_file', 1)))
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
        val_dataset = RawAudioWindowDataset(
            val_file_paths,
            target_sample_rate=sample_rate,
            target_num_samples=target_num_samples,
            min_pitch_shift_semitones=0.0,
            max_pitch_shift_semitones=0.0,
            examples_per_file=1,
            crop_strategy='non_overlapping',
        ) if val_file_paths else None

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
        print(
            'Jukebox raw audio augmentation: '
            f'examples_per_file={examples_per_file}, target_num_samples={target_num_samples}, '
            f'downmix={bool(downmix_cfg.get("enabled", True))}, pitch_enabled={pitch_enabled}, '
            f'pitch_choices={pitch_choices if pitch_choices else "continuous"}, pitch_range={pitch_range}'
        )
        return train_dataset, val_dataset, train_file_paths, val_file_paths, batch_preprocessor


class JukeboxTransformerPriorDataModule:
    """!
    @brief Data module for Jukebox Transformer priors over precomputed indices.
    """

    def __init__(self, config: dict, level: str):
        self.config = config
        self.level = str(level).strip().lower()

    def setup(self, device: torch.device) -> DataBundle:
        dataset_cfg = self.config['dataset']
        training_cfg = self.config['training']
        selected_level = self.level
        target_time_frames = int(dataset_cfg.get('target_time_frames', 2048))
        level_target_time_frames = dataset_cfg.get('level_target_time_frames') or {}
        quantized_path = dataset_cfg.get('quantized_data_path', './data/processed/maestro_quantized/')
        sample_rate = int(dataset_cfg.get('sample_rate', 22050))
        hop_length = int(dataset_cfg.get('hop_length', 256))
        seed = int(training_cfg.get('seed', 42))

        all_file_paths = list_npy_files(dataset_cfg.get('processed_path', ''))
        if not all_file_paths:
            audio_cfg = dataset_cfg.get('audio', {})
            raw_path = dataset_cfg.get('raw_path')
            all_file_paths = list_audio_files(raw_path, extensions=audio_cfg.get('extensions')) if raw_path else []
        if not all_file_paths:
            raise ValueError(
                f"No .npy files found under dataset.processed_path={dataset_cfg.get('processed_path')} "
                f"and no audio files found under dataset.raw_path={dataset_cfg.get('raw_path')}"
            )

        official_train_paths, val_file_paths, test_file_paths = split_paths_by_maestro_metadata(
            all_file_paths,
            dataset_cfg,
        )
        train_file_paths = sorted(official_train_paths + test_file_paths)
        print(
            f'Using metadata splits for {selected_level} prior: '
            f'train={len(official_train_paths)}, test={len(test_file_paths)} combined, '
            f'validation={len(val_file_paths)}'
        )

        train_dataset = JukeboxQuantizedDataset(
            quantized_path=quantized_path,
            file_paths=train_file_paths,
            target_time_frames=target_time_frames,
            level_target_time_frames=level_target_time_frames,
            selected_level=selected_level,
            sample_rate=sample_rate,
            hop_length=hop_length,
        )
        val_dataset = JukeboxQuantizedDataset(
            quantized_path=quantized_path,
            file_paths=val_file_paths,
            target_time_frames=target_time_frames,
            level_target_time_frames=level_target_time_frames,
            selected_level=selected_level,
            sample_rate=sample_rate,
            hop_length=hop_length,
        ) if val_file_paths else None

        train_window_parity_mode = dataset_cfg.get('train_window_parity_mode', 'alternate')
        validation_window_parity = dataset_cfg.get('validation_window_parity', 'even')
        train_sampler = WindowParityEpochSampler(train_dataset, mode=train_window_parity_mode, seed=seed)
        val_loader = None
        val_subset = None
        if val_dataset is not None:
            if hasattr(val_dataset, 'indices_for_window_parity'):
                val_indices = val_dataset.indices_for_window_parity(validation_window_parity)
            else:
                val_indices = list(range(len(val_dataset)))
            val_subset = Subset(val_dataset, val_indices)
            val_loader = DataLoader(
                val_subset,
                batch_size=int(training_cfg['batch_size']),
                shuffle=False,
                **dataloader_kwargs(training_cfg),
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=int(training_cfg['batch_size']),
            sampler=train_sampler,
            **dataloader_kwargs(training_cfg),
        )
        if hasattr(train_dataset, 'window_parity_counts'):
            counts = train_dataset.window_parity_counts()
            print(
                f'Training window parity mode: {train_window_parity_mode} '
                f"(all={counts['all']}, even={counts['even']}, odd={counts['odd']})"
            )
        if val_subset is not None:
            print(
                f'Validation window parity: {validation_window_parity} '
                f'(active={len(val_subset)}, full={len(val_dataset)})'
            )

        metadata = self._infer_transformer_metadata(
            train_dataset,
            selected_level,
            level_target_time_frames,
            target_time_frames,
            sample_rate,
            hop_length,
            train_window_parity_mode,
            seed,
        )
        return DataBundle(
            train_loader=train_loader,
            val_loader=val_loader,
            train_dataset=train_dataset,
            val_dataset=val_subset,
            metadata=metadata,
        )

    def _infer_transformer_metadata(
        self,
        dataset,
        selected_level,
        level_target_time_frames,
        target_time_frames,
        sample_rate,
        hop_length,
        train_window_parity_mode,
        seed,
    ):
        sample = dataset[0]
        target_indices, cond_indices, second_cond_indices, _timing = sample
        top_rows, top_cols = dataset.top_grid
        middle_rows, middle_cols = dataset.middle_grid
        bottom_rows, bottom_cols = dataset.bottom_grid
        grid_shapes = {
            'top': (int(top_rows), int(top_cols)),
            'middle': (int(middle_rows), int(middle_cols)),
            'bottom': (int(bottom_rows), int(bottom_cols)),
        }
        seq_lens = {level: int(rows * cols) for level, (rows, cols) in grid_shapes.items()}
        target_seq_len = int(target_indices.numel())
        if target_seq_len != seq_lens[selected_level]:
            raise ValueError(
                f'{selected_level} target has {target_seq_len} tokens, '
                f'but grid {grid_shapes[selected_level]} implies {seq_lens[selected_level]}.'
            )

        selected_tf = int((level_target_time_frames or {}).get(selected_level, target_time_frames))
        timing_window_seconds = (selected_tf * hop_length) / sample_rate
        cond_seq_len = int(cond_indices.numel()) if cond_indices is not None else 0
        second_cond_seq_len = int(second_cond_indices.numel()) if second_cond_indices is not None else 0
        upsample_stride = None
        cond_block_len = None
        second_upsample_stride = None
        second_cond_block_len = None
        if cond_seq_len > 0:
            upsample_stride, cond_block_len = _compute_2d_stride_from_tensors(
                target_indices,
                cond_indices,
                f'{selected_level} primary conditioning',
            )
        if second_cond_seq_len > 0:
            second_upsample_stride, second_cond_block_len = _compute_2d_stride_from_tensors(
                target_indices,
                second_cond_indices,
                f'{selected_level} secondary conditioning',
            )

        max_train_examples = _max_train_examples_per_epoch(dataset, train_window_parity_mode, seed)
        max_train_batches = int(np.ceil(max_train_examples / int(self.config['training']['batch_size'])))
        grad_accum_steps = int(self.config['training'].get('gradient_accumulation_steps', 1))
        return {
            'family': 'jukebox_transformer_prior',
            'level': selected_level,
            'target_seq_len': target_seq_len,
            'seq_lens': seq_lens,
            'grid_shapes': grid_shapes,
            'cond_seq_len': cond_seq_len,
            'second_cond_seq_len': second_cond_seq_len,
            'cond_time_cols': int(cond_indices.shape[0]) if cond_seq_len and cond_indices.ndim == 2 else 0,
            'cond_freq_bins': int(cond_indices.shape[1]) if cond_seq_len and cond_indices.ndim == 2 else 0,
            'second_cond_time_cols': int(second_cond_indices.shape[0])
            if second_cond_seq_len and second_cond_indices.ndim == 2 else 0,
            'second_cond_freq_bins': int(second_cond_indices.shape[1])
            if second_cond_seq_len and second_cond_indices.ndim == 2 else 0,
            'upsample_stride': upsample_stride,
            'cond_block_len': cond_block_len,
            'second_upsample_stride': second_upsample_stride,
            'second_cond_block_len': second_cond_block_len,
            'timing_window_seconds': float(timing_window_seconds),
            'optimizer_steps_per_epoch': max(1, int(np.ceil(max_train_batches / max(grad_accum_steps, 1)))),
        }


def _tensor_grid_shape(name: str, tensor: torch.Tensor):
    if tensor is None or tensor.numel() == 0:
        raise ValueError(f'{name} is required to infer 2D conditioner geometry.')
    if tensor.ndim != 2:
        raise ValueError(f'{name} must have shape (time_cols, freq_bins), got {tuple(tensor.shape)}')
    time_cols, freq_bins = int(tensor.shape[0]), int(tensor.shape[1])
    if time_cols <= 0 or freq_bins <= 0:
        raise ValueError(f'{name} has invalid shape {tuple(tensor.shape)}')
    return time_cols, freq_bins


def _compute_2d_stride_from_tensors(target_indices: torch.Tensor, cond_indices: torch.Tensor, label: str):
    target_time, target_freq = _tensor_grid_shape(f'{label} target', target_indices)
    cond_time, cond_freq = _tensor_grid_shape(f'{label} conditioning', cond_indices)
    if target_time % cond_time != 0 or target_freq % cond_freq != 0:
        raise ValueError(
            f'Cannot infer 2D conditioner stride for {label}: '
            f'target_shape=({target_time}, {target_freq}), cond_shape=({cond_time}, {cond_freq})'
        )
    return (target_time // cond_time, target_freq // cond_freq), cond_freq


def _max_train_examples_per_epoch(dataset, mode: str, seed: int) -> int:
    if mode == 'all' or not hasattr(dataset, 'indices_for_window_parity'):
        return len(dataset)
    if mode in {'alternate', 'random_per_song'}:
        return max(
            len(dataset.indices_for_window_parity('even')),
            len(dataset.indices_for_window_parity('odd')),
        )
    indices, _ = _indices_for_train_epoch(dataset, mode, 0, seed)
    return len(indices)
