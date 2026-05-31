from datetime import datetime
import argparse
import gc
import os
import sys

import numpy as np
import torch
import yaml

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.raw_audio_dataset import RawAudioWindowDataset, collate_audio_windows, list_audio_files
from datasets.spectrogram_dataset import MmapSpectrogramDataset
from processing.gpu_audio_augmentation import GPUAudioToMelSpectrogram
from train_scripts.jukebox_utils import split_paths_by_maestro_metadata, split_train_val_paths
from train_scripts.resume_utils import load_resume_artifacts
from train_scripts.train_vq_utils import train_model
from utils import initialize_vqvae_model, load_config, load_maestro, set_global_seed


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def _split_audio_paths(all_file_paths, dataset_cfg, validation_split, seed):
    if dataset_cfg.get('metadata_path'):
        return split_paths_by_maestro_metadata(all_file_paths, dataset_cfg)

    train_file_paths, val_file_paths = split_train_val_paths(
        all_file_paths,
        dataset_cfg,
        validation_split=validation_split,
        seed=seed,
    )
    return train_file_paths, val_file_paths or [], []


if __name__ == "__main__":
    # Optional: faster matmul on Ampere+ GPUs
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Capability:", torch.cuda.get_device_capability(0))
        print("CUDA memory allocated (MB):", round(torch.cuda.memory_allocated(0) / 1024**2, 2))

    parser = argparse.ArgumentParser(description="Train a single-level VQ-VAE.")
    parser.add_argument("--config", type=str, default="./config/config_vqvae.yaml")
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=["spectrogram", "audio"],
        default=None,
        help="Override dataset.input_mode. audio reads raw audio and builds augmented mel spectrograms on GPU.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Resume from a previous model.pth/best_model.pth checkpoint, overriding training.retrain/pretrained_weights_path.",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: ignoring unrecognized arguments: {unknown}")

    # Load configuration
    config_path = args.config
    config = load_config(config_path)
    dataset_cfg = config['dataset']
    train_cfg = config['training']
    model_cfg = config['model']

    seed = train_cfg.get('seed', 42)
    set_global_seed(int(seed))

    # Training parameters from config
    batch_size = train_cfg['batch_size']
    learning_rate = train_cfg['learning_rate']
    epochs = train_cfg['epochs']
    early_stopping_patience = train_cfg.get('early_stopping_patience', 20)
    validation_split = train_cfg.get('validation_split', 0.0)
    grad_accum_steps = train_cfg.get('gradient_accumulation_steps', 1)
    num_workers = train_cfg.get('num_workers', 4)
    pin_memory = train_cfg.get('pin_memory', True)
    persist_workers_cfg = train_cfg.get('persist_workers', True)
    persist_workers = bool(persist_workers_cfg) if persist_workers_cfg is not None else True
    prefetch_factor_cfg = train_cfg.get('prefetch_factor', 4)
    prefetch_factor = int(prefetch_factor_cfg) if prefetch_factor_cfg is not None else None
    data_variance_samples = int(train_cfg.get('data_variance_samples', 1000))
    retrain = bool(train_cfg.get('retrain', False))
    pretrained_weights_path = train_cfg.get('pretrained_weights_path')
    if args.resume_checkpoint:
        retrain = True
        pretrained_weights_path = args.resume_checkpoint
    if retrain and not pretrained_weights_path:
        raise ValueError("training.retrain is true but training.pretrained_weights_path is empty.")

    # Dataset/model settings
    input_mode = str(args.input_mode or dataset_cfg.get('input_mode', 'spectrogram')).strip().lower()
    if input_mode not in ('spectrogram', 'audio'):
        raise ValueError("dataset.input_mode must be either 'spectrogram' or 'audio'")

    raw_audio_path = dataset_cfg.get('raw_path')
    spectrograms_path = dataset_cfg.get('processed_path')
    sample_rate = int(dataset_cfg.get('sample_rate', 22050))
    hop_length = int(dataset_cfg.get('hop_length', 256))
    frame_size = int(dataset_cfg.get('frame_size', 2048))
    target_time_frames = int(dataset_cfg.get('target_time_frames', 256))
    spectrogram_type = str(dataset_cfg.get('spectrogram_type', 'mel')).strip().lower()
    n_mels = int(dataset_cfg.get('n_mels', 256))

    model_save_dir = train_cfg['save_dir']
    model_name = model_cfg['name']

    current_datetime = datetime.now()
    formatted_time = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    # Keep the legacy single-VQ-VAE folder shape: save_dir / timestamp / model.pth
    run_dir = os.path.join(model_save_dir, formatted_time)
    os.makedirs(run_dir, exist_ok=True)

    model_file_path = os.path.join(run_dir, "model.pth")
    config_file_path = os.path.join(run_dir, "config.yaml")

    print(f"Training configuration loaded from {config_path}")
    print(f"Model: {model_name}")
    print(f"Model will be saved to: {model_file_path}")
    print(
        f"Dataset: input_mode={input_mode}, target_time_frames={target_time_frames}, "
        f"spectrogram_type={spectrogram_type}, n_mels={n_mels}, "
        f"sample_rate={sample_rate}, hop_length={hop_length}, frame_size={frame_size}"
    )
    print(
        f"Training parameters: batch_size={batch_size}, grad_accum_steps={grad_accum_steps}, "
        f"learning_rate={learning_rate}, epochs={epochs}, seed={seed}"
    )

    resume_history = {}
    initial_best_metric = None
    if retrain:
        resume_history, initial_best_metric, _ = load_resume_artifacts(
            pretrained_weights_path,
            val_key='val_total',
            train_key='total',
        )
        print(f"Retraining enabled from checkpoint: {pretrained_weights_path}")
        if initial_best_metric is not None:
            print(f"Baseline best metric from previous training: {initial_best_metric:.6f}")
        else:
            print("Baseline best metric unavailable from previous training history.")

    # Ensure testing configuration is present for future loading
    if 'testing' not in config:
        config['testing'] = {}
    if 'weights_file_choice' not in config['testing']:
        config['testing']['weights_file_choice'] = 'model.pth'

    config_to_save = dict(config)
    config_to_save['dataset'] = dict(dataset_cfg)
    config_to_save['dataset']['input_mode'] = input_mode
    config_to_save['training'] = dict(train_cfg)
    config_to_save['training']['seed'] = seed
    config_to_save['training']['retrain'] = retrain
    config_to_save['training']['pretrained_weights_path'] = pretrained_weights_path

    with open(config_file_path, 'w') as f:
        yaml.dump(config_to_save, f)

    batch_preprocessor = None
    data_collate_fn = None
    min_max_values = None
    data_variance = None

    if input_mode == 'spectrogram':
        if not spectrograms_path:
            raise ValueError("dataset.processed_path is required for spectrogram input mode.")

        x_all, _ = load_maestro(spectrograms_path, target_time_frames, debug_print=False)
        print("Input shape: ", x_all.shape)
        print("Data range:", x_all.min(), "to", x_all.max())
        print("Data samples length:", x_all.shape[0])

        # If variance is too small, the loss term (MSE / 2*var) becomes huge.
        data_variance = np.var(x_all)
        print(f"Data variance: {data_variance}")

        if validation_split > 0:
            num_samples = len(x_all)
            num_val = int(num_samples * validation_split)
            num_train = num_samples - num_val

            indices = np.random.permutation(num_samples)
            train_indices = indices[:num_train]
            val_indices = indices[num_train:]

            x_train = MmapSpectrogramDataset(x_all, train_indices)
            x_val = MmapSpectrogramDataset(x_all, val_indices)

            print(f"Data split: {len(x_train)} training, {len(x_val)} validation samples.")
            gc.collect()
        else:
            x_train = MmapSpectrogramDataset(x_all)
            x_val = None
            print(f"Using all {len(x_train)} samples for training.")

        time_frames = x_all.shape[2]
        print(f"Time frames detected: {time_frames}")

    else:
        if not raw_audio_path:
            raise ValueError("dataset.raw_path is required for audio input mode.")

        audio_cfg = dataset_cfg.get('audio', {})
        print(f"Scanning for raw audio files in {raw_audio_path}...")
        all_file_paths = list_audio_files(raw_audio_path, extensions=audio_cfg.get('extensions'))
        if not all_file_paths:
            raise FileNotFoundError(f"No audio files found in {raw_audio_path}")
        print(f"Found {len(all_file_paths)} audio files.")

        train_file_paths, val_file_paths, test_file_paths = _split_audio_paths(
            all_file_paths,
            dataset_cfg,
            validation_split=validation_split,
            seed=int(seed),
        )
        if not train_file_paths:
            raise ValueError("The audio split produced no training files.")

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
        x_train = RawAudioWindowDataset(
            train_file_paths,
            target_sample_rate=sample_rate,
            target_num_samples=target_num_samples,
            min_pitch_shift_semitones=float(pitch_range[0]),
            max_pitch_shift_semitones=float(pitch_range[1]),
            examples_per_file=examples_per_file,
            crop_strategy="random",
        )
        x_val = None
        if val_file_paths:
            x_val = RawAudioWindowDataset(
                val_file_paths,
                target_sample_rate=sample_rate,
                target_num_samples=target_num_samples,
                min_pitch_shift_semitones=0.0,
                max_pitch_shift_semitones=0.0,
                examples_per_file=1,
                crop_strategy="non_overlapping",
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
        )
        data_collate_fn = collate_audio_windows
        data_variance = None
        min_max_values = {}

        print(
            "Raw audio augmentation: "
            f"train_files={len(train_file_paths)}, val_files={len(val_file_paths)}, test_files={len(test_file_paths)}, "
            f"train_windows={len(x_train)}, val_windows={len(x_val) if x_val is not None else 0}, "
            f"examples_per_file={examples_per_file}, target_num_samples={target_num_samples}, "
            f"downmix={bool(downmix_cfg.get('enabled', True))}, pitch_enabled={pitch_enabled}, "
            f"pitch_choices={pitch_choices if pitch_choices else 'continuous'}, pitch_range={pitch_range}"
        )

    # Define VQ-VAE model using the utility function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vqvae_model = initialize_vqvae_model(config_to_save, device)

    train_model(
        vqvae_model,
        x_train,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        data_variance=data_variance,
        save_path=model_file_path,
        early_stopping_patience=early_stopping_patience,
        amp=True,
        grad_accum_steps=grad_accum_steps,
        max_grad_norm=1.0,
        model_config=config_to_save['model'],
        min_max_values=min_max_values,
        x_val=x_val,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persist_workers=persist_workers,
        prefetch_factor=prefetch_factor,
        spectrogram_type=spectrogram_type,
        sample_rate=sample_rate,
        hop_length=hop_length,
        frame_size=frame_size,
        n_mels=n_mels,
        batch_preprocessor=batch_preprocessor,
        collate_fn=data_collate_fn,
        data_variance_samples=data_variance_samples,
        resume_checkpoint_path=pretrained_weights_path if retrain else None,
        resume_history=resume_history,
        initial_best_metric=initial_best_metric,
    )
    best_model_path = os.path.join(run_dir, 'best_model.pth')
    if retrain:
        if os.path.isfile(best_model_path):
            print("Retraining improved over baseline: new best model saved.")
        else:
            print("Retraining did not beat baseline best metric in this run.")
    print("Model training complete. Model saved to:", model_file_path)
