import argparse
import json
import math
import os
import sys
from typing import Iterable, List, Optional

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processing.gpu_audio_augmentation import GPUAudioToMelSpectrogram
from processing.preprocess_quantization import (
    DEFAULT_AUDIO_EXTENSIONS,
    _audio_windows_to_specs,
    _resolve_step_frames,
)
from train_scripts.jukebox_utils import split_paths_by_maestro_metadata, split_train_val_paths
from utils import load_config, load_vqvae_hierarchical_model_wrapper, load_vqvae_model
from windowed_data_utils import build_level_starts, build_timing_tensor, source_stem


PIXELCNN_MANIFEST = 'pixelcnn_quantized_manifest.jsonl'
PIXELCNN_CONFIG = 'pixelcnn_quantization_config.json'
FORMAT_NAME = 'single_vqvae_pixelcnn_v1'
TWO_LEVEL_FORMAT_NAME = 'two_level_vqvae_pixelcnn_v1'


def _list_audio_files(source_path: str, extensions: Iterable[str]) -> List[str]:
    extensions = tuple(str(ext).lower() for ext in extensions)
    paths = []
    for root, _, names in os.walk(source_path):
        for name in names:
            if name.lower().endswith(extensions):
                paths.append(os.path.join(root, name))
    return sorted(paths)


def _audio_total_frames(file_path: str, sample_rate: int, hop_length: int) -> int:
    info = sf.info(file_path)
    return max(1, int(math.floor(int(info.frames) * sample_rate / int(info.samplerate) / hop_length)) + 1)


def _build_audio_preprocessor(
    sample_rate: int,
    target_time_frames: int,
    frame_size: int,
    hop_length: int,
    n_mels: int,
    device: torch.device,
) -> GPUAudioToMelSpectrogram:
    preprocessor = GPUAudioToMelSpectrogram(
        sample_rate=sample_rate,
        target_time_frames=target_time_frames,
        n_fft=frame_size,
        hop_length=hop_length,
        n_mels=n_mels,
        random_downmix=False,
        pitch_shift_enabled=False,
    ).to(device)
    preprocessor.eval()
    return preprocessor


def _split_paths(file_paths: List[str], dataset_cfg: dict, validation_split: float, seed: int):
    metadata_path = dataset_cfg.get('metadata_path')
    if metadata_path and os.path.isfile(os.path.expanduser(metadata_path)):
        train_paths, val_paths, test_paths = split_paths_by_maestro_metadata(file_paths, dataset_cfg)
    else:
        train_paths, val_paths = split_train_val_paths(
            file_paths,
            dataset_cfg,
            validation_split=validation_split,
            seed=seed,
        )
        val_paths = val_paths or []
        test_paths = []

    split_by_path = {}
    for path in train_paths:
        split_by_path[os.path.abspath(path)] = 'train'
    for path in val_paths:
        split_by_path[os.path.abspath(path)] = 'validation'
    for path in test_paths:
        split_by_path[os.path.abspath(path)] = 'test'
    return split_by_path, {'train': len(train_paths), 'validation': len(val_paths), 'test': len(test_paths)}


def _encode_specs(model, specs: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(specs).unsqueeze(1).float().to(device)
    with torch.no_grad():
        z_e = model.encoder(x)
        _, indices, _, _, _ = model.vq(z_e)
    return indices.detach().cpu().to(torch.long)


def _encode_two_level_specs(model, specs: np.ndarray, device: torch.device):
    x = torch.from_numpy(specs).unsqueeze(1).float().to(device)
    with torch.no_grad():
        enc_bottom = model.encoder_bottom(x)
        enc_top = model.encoder_top(enc_bottom)
        quant_top = model.pre_vq_conv_top(enc_top)
        quant_bottom = model.pre_vq_conv_bottom(enc_bottom)
        _, top_indices, _, _, _ = model.vq_top(quant_top)
        _, bottom_indices, _, _, _ = model.vq_bottom(quant_bottom)
    return (
        top_indices.detach().cpu().to(torch.long),
        bottom_indices.detach().cpu().to(torch.long),
    )


def _write_payload(
    output_dir: str,
    manifest_file,
    indices: torch.Tensor,
    source_path: str,
    split: str,
    start_frame: int,
    total_frames: int,
    sample_rate: int,
    hop_length: int,
    target_time_frames: int,
):
    stem = source_stem(source_path)
    filename = f'{stem}__start_{int(start_frame):08d}_vqvae_indices.pt'
    payload = {
        'format': FORMAT_NAME,
        'indices': indices.clone(),
        'source_path': source_path,
        'source_basename': os.path.basename(source_path),
        'source_stem': stem,
        'split': split,
        'start_frame': int(start_frame),
        'total_frames': int(total_frames),
        'target_time_frames': int(target_time_frames),
        'timing': build_timing_tensor(
            start_frame=int(start_frame),
            total_frames=int(total_frames),
            sample_rate=int(sample_rate),
            hop_length=int(hop_length),
        ),
    }
    torch.save(payload, os.path.join(output_dir, filename))
    manifest_file.write(
        json.dumps(
            {
                'file': filename,
                'source_basename': payload['source_basename'],
                'source_stem': stem,
                'split': split,
                'start_frame': int(start_frame),
                'total_frames': int(total_frames),
                'target_time_frames': int(target_time_frames),
                'indices_shape': list(indices.shape),
            }
        ) + '\n'
    )


def _write_two_level_payload(
    output_dir: str,
    manifest_file,
    top_indices: torch.Tensor,
    bottom_indices: torch.Tensor,
    source_path: str,
    split: str,
    start_frame: int,
    total_frames: int,
    sample_rate: int,
    hop_length: int,
    target_time_frames: int,
):
    stem = source_stem(source_path)
    filename = f'{stem}__start_{int(start_frame):08d}_two_level_vqvae_indices.pt'
    payload = {
        'format': TWO_LEVEL_FORMAT_NAME,
        'top_indices': top_indices.clone(),
        'bottom_indices': bottom_indices.clone(),
        'source_path': source_path,
        'source_basename': os.path.basename(source_path),
        'source_stem': stem,
        'split': split,
        'start_frame': int(start_frame),
        'total_frames': int(total_frames),
        'target_time_frames': int(target_time_frames),
        'timing': build_timing_tensor(
            start_frame=int(start_frame),
            total_frames=int(total_frames),
            sample_rate=int(sample_rate),
            hop_length=int(hop_length),
        ),
    }
    torch.save(payload, os.path.join(output_dir, filename))
    manifest_file.write(
        json.dumps(
            {
                'file': filename,
                'source_basename': payload['source_basename'],
                'source_stem': stem,
                'split': split,
                'start_frame': int(start_frame),
                'total_frames': int(total_frames),
                'target_time_frames': int(target_time_frames),
                'top_indices_shape': list(top_indices.shape),
                'bottom_indices_shape': list(bottom_indices.shape),
            }
        ) + '\n'
    )


def precompute_single_vqvae_indices(
    vqvae_model_path: str,
    source_path: str,
    output_dir: str,
    weights_file: Optional[str],
    batch_size: int,
    target_time_frames: int,
    step_frames: int,
    sample_rate: int,
    hop_length: int,
    frame_size: int,
    n_mels: int,
    audio_extensions: Iterable[str],
    validation_split: float,
    seed: int,
    max_files: Optional[int],
    device: torch.device,
):
    os.makedirs(output_dir, exist_ok=True)

    vqvae = load_vqvae_model(vqvae_model_path, device, weights_file=weights_file).eval()
    dataset_cfg = load_config(_resolve_config_path(vqvae_model_path)).get('dataset', {})
    dataset_cfg = dict(dataset_cfg)
    dataset_cfg['raw_path'] = source_path

    file_paths = _list_audio_files(source_path, audio_extensions)

    if max_files is not None:
        file_paths = file_paths[: int(max_files)]
    if not file_paths:
        raise FileNotFoundError(f'No audio files found in {source_path} with extensions={audio_extensions}')

    split_by_path, split_counts = _split_paths(file_paths, dataset_cfg, validation_split, seed)

    config_payload = {
        'format': FORMAT_NAME,
        'source_mode': 'audio',
        'source_path': source_path,
        'output_dir': output_dir,
        'vqvae_model_path': vqvae_model_path,
        'weights_file': weights_file,
        'batch_size': int(batch_size),
        'target_time_frames': int(target_time_frames),
        'step_frames': int(step_frames),
        'sample_rate': int(sample_rate),
        'hop_length': int(hop_length),
        'frame_size': int(frame_size),
        'n_mels': int(n_mels),
        'audio_extensions': list(audio_extensions),
        'validation_split': float(validation_split),
        'seed': int(seed),
        'max_files': None if max_files is None else int(max_files),
        'num_embeddings': int(vqvae.vq.num_embeddings),
        'split_counts_files': split_counts,
    }
    with open(os.path.join(output_dir, PIXELCNN_CONFIG), 'w', encoding='utf-8') as f:
        json.dump(config_payload, f, indent=2)

    preprocessor = _build_audio_preprocessor(
        sample_rate=sample_rate,
        target_time_frames=target_time_frames,
        frame_size=frame_size,
        hop_length=hop_length,
        n_mels=n_mels,
        device=device,
    )

    total_examples = 0
    total_frames_by_path = {}
    starts_by_path = {}
    for file_path in file_paths:
        total_frames = _audio_total_frames(file_path, sample_rate, hop_length)
        starts = build_level_starts(total_frames, target_time_frames, step_frames)
        total_frames_by_path[file_path] = total_frames
        starts_by_path[file_path] = starts
        total_examples += len(starts)

    print(
        'Precomputing single VQ-VAE PixelCNN indices '
        f'(files={len(file_paths)}, examples={total_examples}, '
        f'target_time_frames={target_time_frames}, step_frames={step_frames}, batch_size={batch_size})...'
    )
    print(
        'File splits: '
        f"train={split_counts['train']}, validation={split_counts['validation']}, test={split_counts['test']}"
    )

    examples_by_split = {'train': 0, 'validation': 0, 'test': 0, 'unknown': 0}
    manifest_path = os.path.join(output_dir, PIXELCNN_MANIFEST)
    with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
        progress = tqdm(total=total_examples, desc='Quantizing Single VQ-VAE Windows')

        for file_path in file_paths:
            total_frames = total_frames_by_path[file_path]
            starts = starts_by_path[file_path]
            split = split_by_path.get(os.path.abspath(file_path), 'unknown')

            for batch_start in range(0, len(starts), batch_size):
                batch_starts = starts[batch_start : batch_start + batch_size]
                batch_specs = _audio_windows_to_specs(
                    file_path=file_path,
                    starts=batch_starts,
                    window_time_frames=target_time_frames,
                    preprocessor=preprocessor,
                    sample_rate=sample_rate,
                    hop_length=hop_length,
                    device=device,
                )

                batch_indices = _encode_specs(vqvae, batch_specs, device)

                for item_idx, start_frame in enumerate(batch_starts):
                    _write_payload(
                        output_dir=output_dir,
                        manifest_file=manifest_file,
                        indices=batch_indices[item_idx],
                        source_path=file_path,
                        split=split,
                        start_frame=int(start_frame),
                        total_frames=total_frames,
                        sample_rate=sample_rate,
                        hop_length=hop_length,
                        target_time_frames=target_time_frames,
                    )
                    examples_by_split[split] = examples_by_split.get(split, 0) + 1

                progress.update(len(batch_starts))

            if device.type == 'cuda':
                torch.cuda.empty_cache()

        progress.close()

    config_payload['split_counts_examples'] = examples_by_split
    with open(os.path.join(output_dir, PIXELCNN_CONFIG), 'w', encoding='utf-8') as f:
        json.dump(config_payload, f, indent=2)

    print('\nQuantization preprocessing complete.')
    print(f'Output directory: {output_dir}')
    print(f'Manifest: {manifest_path}')
    print(f'Example splits: {examples_by_split}')


def precompute_two_level_vqvae_indices(
    vqvae_model_path: str,
    source_path: str,
    output_dir: str,
    weights_file: Optional[str],
    batch_size: int,
    target_time_frames: int,
    step_frames: int,
    sample_rate: int,
    hop_length: int,
    frame_size: int,
    n_mels: int,
    audio_extensions: Iterable[str],
    validation_split: float,
    seed: int,
    max_files: Optional[int],
    device: torch.device,
):
    os.makedirs(output_dir, exist_ok=True)

    model_ref = (
        os.path.join(vqvae_model_path, weights_file)
        if weights_file and os.path.isdir(vqvae_model_path)
        else vqvae_model_path
    )
    vqvae = load_vqvae_hierarchical_model_wrapper(model_ref, device).eval()
    dataset_cfg = load_config(_resolve_config_path(vqvae_model_path)).get('dataset', {})
    dataset_cfg = dict(dataset_cfg)
    dataset_cfg['raw_path'] = source_path

    file_paths = _list_audio_files(source_path, audio_extensions)

    if max_files is not None:
        file_paths = file_paths[: int(max_files)]
    if not file_paths:
        raise FileNotFoundError(f'No audio files found in {source_path} with extensions={audio_extensions}')

    split_by_path, split_counts = _split_paths(file_paths, dataset_cfg, validation_split, seed)

    config_payload = {
        'format': TWO_LEVEL_FORMAT_NAME,
        'source_mode': 'audio',
        'source_path': source_path,
        'output_dir': output_dir,
        'vqvae_model_path': vqvae_model_path,
        'weights_file': weights_file,
        'batch_size': int(batch_size),
        'target_time_frames': int(target_time_frames),
        'step_frames': int(step_frames),
        'sample_rate': int(sample_rate),
        'hop_length': int(hop_length),
        'frame_size': int(frame_size),
        'n_mels': int(n_mels),
        'audio_extensions': list(audio_extensions),
        'validation_split': float(validation_split),
        'seed': int(seed),
        'max_files': None if max_files is None else int(max_files),
        'num_embeddings': [int(vqvae.vq_top.num_embeddings), int(vqvae.vq_bottom.num_embeddings)],
        'num_embeddings_top': int(vqvae.vq_top.num_embeddings),
        'num_embeddings_bottom': int(vqvae.vq_bottom.num_embeddings),
        'split_counts_files': split_counts,
    }
    with open(os.path.join(output_dir, PIXELCNN_CONFIG), 'w', encoding='utf-8') as f:
        json.dump(config_payload, f, indent=2)

    preprocessor = _build_audio_preprocessor(
        sample_rate=sample_rate,
        target_time_frames=target_time_frames,
        frame_size=frame_size,
        hop_length=hop_length,
        n_mels=n_mels,
        device=device,
    )

    total_examples = 0
    total_frames_by_path = {}
    starts_by_path = {}
    for file_path in file_paths:
        total_frames = _audio_total_frames(file_path, sample_rate, hop_length)
        starts = build_level_starts(total_frames, target_time_frames, step_frames)
        total_frames_by_path[file_path] = total_frames
        starts_by_path[file_path] = starts
        total_examples += len(starts)

    print(
        'Precomputing two-level VQ-VAE PixelCNN indices '
        f'(files={len(file_paths)}, examples={total_examples}, '
        f'target_time_frames={target_time_frames}, step_frames={step_frames}, batch_size={batch_size})...'
    )
    print(
        'File splits: '
        f"train={split_counts['train']}, validation={split_counts['validation']}, test={split_counts['test']}"
    )

    examples_by_split = {'train': 0, 'validation': 0, 'test': 0, 'unknown': 0}
    manifest_path = os.path.join(output_dir, PIXELCNN_MANIFEST)
    with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
        progress = tqdm(total=total_examples, desc='Quantizing Two-Level VQ-VAE Windows')

        for file_path in file_paths:
            total_frames = total_frames_by_path[file_path]
            starts = starts_by_path[file_path]
            split = split_by_path.get(os.path.abspath(file_path), 'unknown')

            for batch_start in range(0, len(starts), batch_size):
                batch_starts = starts[batch_start : batch_start + batch_size]
                batch_specs = _audio_windows_to_specs(
                    file_path=file_path,
                    starts=batch_starts,
                    window_time_frames=target_time_frames,
                    preprocessor=preprocessor,
                    sample_rate=sample_rate,
                    hop_length=hop_length,
                    device=device,
                )

                top_batch_indices, bottom_batch_indices = _encode_two_level_specs(vqvae, batch_specs, device)

                for item_idx, start_frame in enumerate(batch_starts):
                    _write_two_level_payload(
                        output_dir=output_dir,
                        manifest_file=manifest_file,
                        top_indices=top_batch_indices[item_idx],
                        bottom_indices=bottom_batch_indices[item_idx],
                        source_path=file_path,
                        split=split,
                        start_frame=int(start_frame),
                        total_frames=total_frames,
                        sample_rate=sample_rate,
                        hop_length=hop_length,
                        target_time_frames=target_time_frames,
                    )
                    examples_by_split[split] = examples_by_split.get(split, 0) + 1

                progress.update(len(batch_starts))

            if device.type == 'cuda':
                torch.cuda.empty_cache()

        progress.close()

    config_payload['split_counts_examples'] = examples_by_split
    with open(os.path.join(output_dir, PIXELCNN_CONFIG), 'w', encoding='utf-8') as f:
        json.dump(config_payload, f, indent=2)

    print('\nTwo-level quantization preprocessing complete.')
    print(f'Output directory: {output_dir}')
    print(f'Manifest: {manifest_path}')
    print(f'Example splits: {examples_by_split}')


def _resolve_config_path(model_path: str) -> str:
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, 'config.yaml')
    else:
        config_path = os.path.join(os.path.dirname(model_path), 'config.yaml')
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f'Config file not found at {config_path}')
    return config_path


def main():
    parser = argparse.ArgumentParser(
        description='Precompute VQ-VAE code indices for PixelCNN training.'
    )
    parser.add_argument(
        '--variant',
        type=str,
        choices=['single', 'two_level', 'hierarchical'],
        default='single',
        help='Use single for VQ-VAE indices or two_level/hierarchical for top+bottom VQ-VAE-2 indices.',
    )
    parser.add_argument('--vqvae_model', type=str, required=True, help='VQ-VAE run directory or checkpoint path.')
    parser.add_argument('--source_path', type=str, default=None, help='Raw audio directory. Defaults to dataset.raw_path from the VQ-VAE config.')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory for quantized index .pt files.')
    parser.add_argument('--weights_file', type=str, default=None, help='Checkpoint filename when --vqvae_model is a directory.')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of windows to quantize per VQ-VAE forward pass.')
    parser.add_argument('--target_time_frames', type=int, default=None, help='Window size in spectrogram frames. Defaults to VQ-VAE config dataset.target_time_frames.')
    parser.add_argument('--step_frames', type=int, default=None, help='Window hop in spectrogram frames. Defaults from --overlap_fraction.')
    parser.add_argument('--overlap_fraction', type=float, default=0.0, help='Overlap used when --step_frames is omitted.')
    parser.add_argument('--sample_rate', type=int, default=None)
    parser.add_argument('--hop_length', type=int, default=None)
    parser.add_argument('--frame_size', type=int, default=None)
    parser.add_argument('--n_mels', type=int, default=None)
    parser.add_argument('--audio_extensions', nargs='+', default=None)
    parser.add_argument('--validation_split', type=float, default=None, help='Fallback split if no metadata CSV exists.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_files', type=int, default=None, help='Optional smoke-test limit.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    vqvae_config = load_config(_resolve_config_path(args.vqvae_model))
    dataset_cfg = vqvae_config.get('dataset', {})
    training_cfg = vqvae_config.get('training', {})

    input_mode = str(dataset_cfg.get('input_mode', 'audio')).strip().lower()
    if input_mode != 'audio':
        raise ValueError("VQ-VAE quantization preprocessing now requires dataset.input_mode='audio'.")

    if args.source_path is None:
        args.source_path = dataset_cfg.get('raw_path')
    if not args.source_path:
        raise ValueError('--source_path is required when dataset.raw_path is missing from the VQ-VAE config.')
    if not os.path.isdir(args.source_path):
        raise FileNotFoundError(f'Source path not found: {args.source_path}')

    target_time_frames = int(args.target_time_frames or dataset_cfg.get('target_time_frames', 256))
    step_frames = _resolve_step_frames(args.step_frames, target_time_frames, args.overlap_fraction, 'single_vqvae')
    sample_rate = int(args.sample_rate or dataset_cfg.get('sample_rate', 22050))
    hop_length = int(args.hop_length or dataset_cfg.get('hop_length', 256))
    frame_size = int(args.frame_size or dataset_cfg.get('frame_size', 2048))
    n_mels = int(args.n_mels or dataset_cfg.get('n_mels', 256))
    audio_cfg = dataset_cfg.get('audio', {})
    audio_extensions = args.audio_extensions or audio_cfg.get('extensions') or list(DEFAULT_AUDIO_EXTENSIONS)
    validation_split = float(
        args.validation_split
        if args.validation_split is not None
        else training_cfg.get('validation_split', 0.2)
    )

    variant = 'two_level' if args.variant == 'hierarchical' else args.variant
    if args.output_dir is None:
        args.output_dir = (
            './data/processed/vqvae_pixelcnn_quantized/'
            if variant == 'single'
            else './data/processed/hierarchical_vqvae_quantized/'
        )

    print(f'Using device: {args.device}')
    print(f'Quantization variant: {variant}')
    print(f'VQ-VAE model: {args.vqvae_model}')
    print(f'Source mode/path: audio / {args.source_path}')
    print(f'Output directory: {args.output_dir}')
    print(
        f'Window config: target_time_frames={target_time_frames}, step_frames={step_frames}, '
        f'overlap_fraction={args.overlap_fraction}'
    )
    print(
        f'Audio/mel config: sample_rate={sample_rate}, hop_length={hop_length}, '
        f'frame_size={frame_size}, n_mels={n_mels}'
    )
    print()

    common_kwargs = {
        'vqvae_model_path': args.vqvae_model,
        'source_path': args.source_path,
        'output_dir': args.output_dir,
        'weights_file': args.weights_file,
        'batch_size': max(1, int(args.batch_size)),
        'target_time_frames': target_time_frames,
        'step_frames': step_frames,
        'sample_rate': sample_rate,
        'hop_length': hop_length,
        'frame_size': frame_size,
        'n_mels': n_mels,
        'audio_extensions': audio_extensions,
        'validation_split': validation_split,
        'seed': int(args.seed),
        'max_files': args.max_files,
        'device': torch.device(args.device),
    }
    if variant == 'single':
        precompute_single_vqvae_indices(**common_kwargs)
    else:
        precompute_two_level_vqvae_indices(**common_kwargs)


if __name__ == '__main__':
    main()
