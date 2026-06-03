from datetime import datetime
import argparse
import torch
import numpy as np
import os
import yaml
import sys
import pickle
import gc
import glob
import torch.nn as nn

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling.torch.jukebox_vq_vae import JukeboxVQVAE
from generation.generate import *
from utils import set_global_seed, load_config, compute_dataset_variance, compute_small_sample_variance
from datasets.spectrogram_dataset import LazySpectrogramDataset
from datasets.raw_audio_dataset import RawAudioWindowDataset, collate_audio_windows, list_audio_files
from processing.gpu_audio_augmentation import GPUAudioToMelSpectrogram
from training.jukebox_vqvae import train_vqvae_jukebox
from train_scripts.jukebox_utils import split_paths_by_maestro_metadata
from train_scripts.resume_utils import load_resume_artifacts

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


if __name__ == "__main__":
    # Optional: faster matmul on Ampere+ GPUs
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Capability:", torch.cuda.get_device_capability(0))
        print("CUDA memory allocated (MB):", round(torch.cuda.memory_allocated(0)/1024**2, 2))

    parser = argparse.ArgumentParser(description="Train Jukebox-style VQ-VAE level model")
    parser.add_argument(
        "--level",
        type=str,
        choices=["bottom", "middle", "top"],
        default=None,
        help="Override config model.selected_level.",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=["spectrogram", "audio"],
        default=None,
        help="Override dataset.input_mode. 'audio' reads raw MAESTRO audio and builds mel spectrograms on GPU.",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: ignoring unrecognized arguments: {unknown}")

    # Load configuration
    config_path = "./config/config_jukebox.yaml"
    config = load_config(config_path)

    # Determine selected level first
    model_cfg = config['model']
    selected_level = args.level or model_cfg.get('selected_level', 'bottom')
    selected_level = str(selected_level).lower()

    # Merge general training config with level-specific config
    train_general_cfg = config['training']
    train_level_cfg = train_general_cfg.get(selected_level, {})
    # Filter out level-specific keys and merge
    train_cfg = {k: v for k, v in train_general_cfg.items() if k not in ['bottom', 'middle', 'top']}
    train_cfg.update(train_level_cfg)
    seed = int(train_cfg.get('seed', 42))
    set_global_seed(seed)

    # Extract parameters
    batch_size = train_cfg.get('batch_size')
    grad_accum_steps = train_cfg.get('gradient_accumulation_steps', 1)
    learning_rate = train_cfg.get('learning_rate')
    epochs = train_cfg.get('epochs')
    early_stopping_patience = train_cfg.get('early_stopping_patience', 20)
    num_workers = train_cfg.get('num_workers', 4)
    pin_memory = train_cfg.get('pin_memory', True)
    persist_workers_cfg = train_cfg.get('persist_workers', True)
    persist_workers = bool(persist_workers_cfg) if persist_workers_cfg is not None else True
    prefetch_factor_cfg = train_cfg.get('prefetch_factor', 4)
    prefetch_factor = int(prefetch_factor_cfg) if prefetch_factor_cfg is not None else None

    # Dataset and model paths
    dataset_cfg = config['dataset']
    input_mode = str(args.input_mode or dataset_cfg.get('input_mode', 'spectrogram')).strip().lower()
    if input_mode not in ('spectrogram', 'audio'):
        raise ValueError("dataset.input_mode must be either 'spectrogram' or 'audio'")
    raw_audio_path = dataset_cfg.get('raw_path')
    spectrograms_path = dataset_cfg['processed_path']
    min_max_values_path = dataset_cfg.get('min_max_values_path')
    sample_rate = int(dataset_cfg.get('sample_rate', 22050))
    hop_length = int(dataset_cfg.get('hop_length', 256))
    frame_size = int(dataset_cfg.get('frame_size', 512))
    spectrogram_type_cfg = dataset_cfg.get('spectrogram_type')
    if spectrogram_type_cfg is None:
        spectrogram_type = 'mel' if 'mel' in str(spectrograms_path).lower() else 'linear'
    else:
        spectrogram_type = str(spectrogram_type_cfg).strip().lower()
    n_mels = int(dataset_cfg.get('n_mels', 256))
    model_save_dir = train_cfg['save_dir']
    model_name = model_cfg['name']
    retrain = bool(train_cfg.get('retrain', False))
    pretrained_weights_path = train_cfg.get('pretrained_weights_path')

    # Individual level parameters
    level_profiles = model_cfg.get('level_profiles')
    if level_profiles is None:
        raise ValueError("model.level_profiles is missing from config. It should define parameters for each level (bottom, middle, top).")
    if selected_level not in level_profiles:
        available = ', '.join(level_profiles.keys())
        raise ValueError(f"Invalid selected level '{selected_level}'. Available levels: {available}")

    selected_profile = level_profiles[selected_level]
    hidden_dim = selected_profile.get('hidden_dim')
    levels = selected_profile.get('levels')
    num_residual_layers = selected_profile.get('num_residual_layers', 4)
    if levels is None:
        raise ValueError(f"Missing 'levels' for profile '{selected_level}' in model.level_profiles")

    # Use the level-specific target_time_frames if defined in level_profiles; otherwise
    # fall back to the dataset-level default. This ensures Bottom trains on short clips
    # (≈1.5s), Middle on medium clips (≈6s), and Top on the full long window (≈24s).
    _profile_tf = (level_profiles.get(selected_level) or {}).get('target_time_frames')
    target_time_frames = int(_profile_tf if _profile_tf is not None else dataset_cfg.get('target_time_frames', 2048))

    current_datetime = datetime.now()
    formatted_time = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    print(f"Configuration loaded. Model: {model_name}, Level: {selected_level}, Save Dir: {model_save_dir}")
    print(f"Training parameters: batch_size={batch_size}, grad_accum_steps={grad_accum_steps}, learning_rate={learning_rate}, epochs={epochs}, early_stopping_patience={early_stopping_patience}")
    print(f"Reproducibility seed: {seed}")
    print(f"Data loading parameters: num_workers={num_workers}, pin_memory={pin_memory}, persist_workers={persist_workers}, prefetch_factor={prefetch_factor}")
    print(f"Dataset target_time_frames={target_time_frames}, input_mode={input_mode}, split_source=metadata_csv")
    print(f"Audio inversion settings: spectrogram_type={spectrogram_type}, sample_rate={sample_rate}, hop_length={hop_length}, frame_size={frame_size}, n_mels={n_mels}")

    # structure: model_save_dir / formatted_time / model.pth
    run_dir = os.path.join(model_save_dir, f"{model_name}_{selected_level}", formatted_time)
    os.makedirs(run_dir, exist_ok=True)

    model_file_path = os.path.join(run_dir, "model.pth")
    config_file_path = os.path.join(run_dir, "config.yaml")

    print(f"Training configuration loaded from {config_path}")
    print(f"Selected Jukebox level: {selected_level} (levels={levels}, residual_layers={num_residual_layers})")
    print(f"Model will be saved to: {model_file_path}")

    resume_history = {}
    initial_best_metric = None
    if retrain:
        if not pretrained_weights_path:
            raise ValueError("training.retrain is true but training.pretrained_weights_path is empty.")
        resume_history, initial_best_metric, _ = load_resume_artifacts(pretrained_weights_path, val_key='val_total', train_key='total')
        print(f"Retraining enabled from checkpoint: {pretrained_weights_path}")
        if initial_best_metric is not None:
            print(f"Baseline best metric from previous training: {initial_best_metric:.6f}")
        else:
            print("Baseline best metric unavailable from previous training history.")

    # Save the config used for this training run immediately
    config_to_save = dict(config)
    config_to_save['training'] = dict(config['training'])
    config_to_save['training']['retrain'] = retrain
    config_to_save['training']['pretrained_weights_path'] = pretrained_weights_path
    config_to_save['model'] = dict(config['model'])
    config_to_save['model']['selected_level'] = selected_level
    config_to_save['dataset'] = dict(config['dataset'])

    with open(config_file_path, 'w') as f:
        yaml.dump(config_to_save, f)
    
    batch_preprocessor = None
    data_collate_fn = None
    min_max_values = None
    data_variance = None
    data_variance_samples = int(train_cfg.get('data_variance_samples', 1000))

    if input_mode == 'spectrogram':
        print(f"Scanning for spectrogram files in {spectrograms_path}...")
        all_file_paths = glob.glob(os.path.join(spectrograms_path, "**/*.npy"), recursive=True)
        all_file_paths = sorted(all_file_paths)

        if not all_file_paths:
            raise FileNotFoundError(f"No .npy files found in {spectrograms_path}")

        print(f"Found {len(all_file_paths)} files. Creating lazy datasets...")
        train_file_paths, val_file_paths, test_file_paths = split_paths_by_maestro_metadata(all_file_paths, dataset_cfg)

        data_variance = compute_small_sample_variance(train_file_paths, samples=data_variance_samples)
        print(f"Computed train-set sample variance ({data_variance_samples} samples max): {data_variance:.6f}")
        gc.collect()

        x_train = LazySpectrogramDataset(
            train_file_paths,
            target_time_frames=target_time_frames,
            crop_strategy="random",
        )
        x_val = LazySpectrogramDataset(
            val_file_paths,
            target_time_frames=target_time_frames,
            crop_strategy="non_overlapping",
        )

        if not min_max_values_path:
            raise ValueError("dataset.min_max_values_path is required for spectrogram input mode.")
        with open(min_max_values_path, "rb") as f:
            min_max_values = pickle.load(f)

    else:
        if not raw_audio_path:
            raise ValueError("dataset.raw_path is required for audio input mode.")

        audio_cfg = dataset_cfg.get('audio', {})
        print(f"Scanning for raw audio files in {raw_audio_path}...")
        all_file_paths = list_audio_files(raw_audio_path, extensions=audio_cfg.get('extensions'))
        if not all_file_paths:
            raise FileNotFoundError(f"No audio files found in {raw_audio_path}")
        print(f"Found {len(all_file_paths)} audio files. Creating raw-audio datasets...")

        train_file_paths, val_file_paths, test_file_paths = split_paths_by_maestro_metadata(all_file_paths, dataset_cfg)

        examples_by_level = audio_cfg.get('examples_per_file_by_level', {})
        examples_per_file = int(examples_by_level.get(selected_level, audio_cfg.get('examples_per_file', 1)))

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
            f"examples_per_file={examples_per_file}, "
            f"target_num_samples={target_num_samples}, "
            f"downmix={bool(downmix_cfg.get('enabled', True))}, "
            f"pitch_enabled={pitch_enabled}, "
            f"pitch_choices={pitch_choices if pitch_choices else 'continuous'}, "
            f"pitch_range={pitch_range}"
        )

    print(f"Data split samples/windows: {len(x_train)} training, {len(x_val)} validation.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    activation_name = str(model_cfg.get('activation', '')).lower()
    activation_layer = nn.Sigmoid() if activation_name == 'sigmoid' else None

    # Assert model config has all paramters needed for model initialization
    required_params = ['input_channels', 'num_embeddings', 'embedding_dim', 'beta', 'conv_type', 'dilation_growth_rate', 'channel_growth', 'ema_decay', 'epsilon', 'restart_threshold']
    missing_params = [p for p in required_params if p not in model_cfg]
    if missing_params:
        raise ValueError(f"Missing required model config parameters: {', '.join(missing_params)}")

    jukebox_model = JukeboxVQVAE(
        input_channels=model_cfg['input_channels'],
        hidden_dim=hidden_dim,
        levels=levels,
        num_residual_layers=num_residual_layers,
        num_embeddings=model_cfg.get('num_embeddings'),
        embedding_dim=model_cfg.get('embedding_dim'),
        beta=model_cfg.get('beta'),
        conv_type=model_cfg.get('conv_type'),
        activation_layer=activation_layer,
        dilation_growth_rate=model_cfg.get('dilation_growth_rate'),
        channel_growth=model_cfg.get('channel_growth'),
        ema_decay=model_cfg.get('ema_decay'),
        epsilon=model_cfg.get('epsilon'),
        restart_threshold=model_cfg.get('restart_threshold'),
    ).to(device)

    train_vqvae_jukebox(
        model=jukebox_model,
        x_train=x_train,
        train_file_paths=train_file_paths,
        min_max_values=min_max_values,
        device=device,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        learning_rate=learning_rate,
        epochs=epochs,
        save_path=model_file_path,
        data_variance=data_variance,
        early_stopping_patience=early_stopping_patience,
        x_val=x_val,
        val_file_paths=val_file_paths,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persist_workers=persist_workers,
        prefetch_factor=prefetch_factor,
        resume_checkpoint_path=pretrained_weights_path if retrain else None,
        resume_history=resume_history,
        initial_best_metric=initial_best_metric,
        spectrogram_type=spectrogram_type,
        sample_rate=sample_rate,
        hop_length=hop_length,
        frame_size=frame_size,
        n_mels=n_mels,
        batch_preprocessor=batch_preprocessor,
        collate_fn=data_collate_fn,
        data_variance_samples=data_variance_samples,
    )

    best_model_path = os.path.join(run_dir, 'best_model.pth')
    if retrain:
        if os.path.isfile(best_model_path):
            print("Retraining improved over baseline: new best model saved.")
        else:
            print("Retraining did not beat baseline best metric in this run.")
    print("Training completed.")
