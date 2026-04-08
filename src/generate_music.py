import os
import pickle
import sys
import soundfile as sf
import argparse
import random
from datetime import datetime
from typing import Optional, List
import numpy as np

import torch
import matplotlib.pyplot as plt

# Add 'src' to sys.path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.jukebox_hierarchical_quantized_dataset import JukeboxHierarchicalQuantizedDataset
from modeling.torch.transformer_prior_conditioned import TransformerPriorConditioned
from utils import load_maestro, load_config
from train_scripts.jukebox_utils import load_jukebox_model
from test_scripts.test_transformer_prior import load_transformer_prior
from generation.soundgenerator import SoundGenerator
from processing.preprocess_audio import SAMPLE_RATE, HOP_LENGTH


DEFAULT_TOP_RUN_ROOT = "models/transformer_prior/jukebox_maestro2011_target_time_frames_1024_top_transformer_prior"
DEFAULT_MIDDLE_RUN_ROOT = "models/transformer_prior/jukebox_maestro2011_target_time_frames_1024_middle_transformer_prior"
DEFAULT_BOTTOM_RUN_ROOT = "models/transformer_prior/jukebox_maestro2011_target_time_frames_1024_bottom_transformer_prior"


def _prepare_min_max_values(min_max_values: object, count: int) -> list:
    if isinstance(min_max_values, dict):
        values = list(min_max_values.values())
    elif isinstance(min_max_values, list):
        values = min_max_values
    else:
        raise ValueError('min_max_values must be a dict or list')

    if not values:
        raise ValueError('min_max_values is empty')

    if len(values) >= count:
        return values[:count]

    repeats = (count + len(values) - 1) // len(values)
    tiled = (values * repeats)[:count]
    return tiled


def _resolve_latest_config_path(run_root: str, level_name: str) -> str:
    if os.path.isfile(run_root):
        return run_root

    if not os.path.isdir(run_root):
        raise FileNotFoundError(f"{level_name} run root does not exist: {run_root}")

    direct_cfg = os.path.join(run_root, 'config.yaml')
    if os.path.isfile(direct_cfg):
        return direct_cfg

    candidates = []
    for entry in os.listdir(run_root):
        entry_path = os.path.join(run_root, entry)
        if not os.path.isdir(entry_path):
            continue
        cfg_path = os.path.join(entry_path, 'config.yaml')
        if os.path.isfile(cfg_path):
            candidates.append(cfg_path)

    if not candidates:
        raise FileNotFoundError(
            f"No config.yaml found in {run_root} for level {level_name}."
        )

    return max(candidates, key=os.path.getmtime)


def _resolve_prior_config_path(
    explicit_path: Optional[str],
    default_run_root: str,
    level_name: str,
) -> str:
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f"{level_name} config path not found: {explicit_path}")
        if os.path.isdir(explicit_path):
            return _resolve_latest_config_path(explicit_path, level_name)
        return explicit_path

    return _resolve_latest_config_path(default_run_root, level_name)


def _debug(msg: str) -> None:
    print(f'[DEBUG] {msg}')


def _normalize_candidate_path(path: Optional[str]) -> Optional[str]:
    if not path or not isinstance(path, str):
        return None
    return os.path.abspath(os.path.expanduser(path))


def _resolve_min_max_values_path(bottom_config: dict) -> str:
    candidates: List[str] = []

    direct_path = _normalize_candidate_path(bottom_config.get('dataset', {}).get('min_max_values_path'))
    if direct_path:
        candidates.append(direct_path)

    processed_path = _normalize_candidate_path(bottom_config.get('dataset', {}).get('processed_path'))
    if processed_path:
        candidates.append(os.path.join(processed_path, 'min_max_values.pkl'))

    bottom_model_dir = _normalize_candidate_path(bottom_config.get('vqvae', {}).get('bottom_model_dir'))
    if bottom_model_dir:
        bottom_vqvae_cfg = os.path.join(bottom_model_dir, 'config.yaml')
        if os.path.exists(bottom_vqvae_cfg):
            _debug(f'Loading bottom VQ-VAE config for min/max fallback: {bottom_vqvae_cfg}')
            vq_cfg = load_config(bottom_vqvae_cfg)
            vq_direct = _normalize_candidate_path(vq_cfg.get('dataset', {}).get('min_max_values_path'))
            if vq_direct:
                candidates.append(vq_direct)
            vq_processed = _normalize_candidate_path(vq_cfg.get('dataset', {}).get('processed_path'))
            if vq_processed:
                candidates.append(os.path.join(vq_processed, 'min_max_values.pkl'))

    seen = set()
    unique_candidates = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique_candidates.append(path)

    _debug('Min/max candidate paths checked:')
    for path in unique_candidates:
        _debug(f'  - {path}')

    for path in unique_candidates:
        if os.path.exists(path):
            _debug(f'Using min/max path: {path}')
            return path

    raise FileNotFoundError(
        'Could not resolve min_max_values.pkl. Checked dataset.min_max_values_path, '
        'dataset.processed_path/min_max_values.pkl, and bottom VQ-VAE config fallbacks.'
    )


def _generate_level_tokens(
    prior,
    seq_len: int,
    chunks_to_generate: int,
    device: torch.device,
    temperature: float,
    top_k: Optional[int],
    upper_tokens_list: Optional[List[np.ndarray]] = None,
    use_time_id: bool = False,
) -> List[np.ndarray]:
    token_blocks = []
    curr_start_token = None
    max_time_steps = getattr(prior, 'max_time_steps', None) if use_time_id else None

    for chunk in range(chunks_to_generate):
        upper_indices = None
        if upper_tokens_list is not None:
            upper_indices = torch.from_numpy(upper_tokens_list[chunk]).to(device)

        generate_kwargs = {
            'batch_size': 1,
            'start_tokens': curr_start_token,
            'upper_indices': upper_indices,
            'seq_len': seq_len,
            'temperature': temperature,
            'top_k': top_k,
            'device': device,
        }
        if use_time_id:
            time_id_value = chunk
            if isinstance(max_time_steps, int) and max_time_steps > 0:
                time_id_value = chunk % max_time_steps
            current_time_id = torch.tensor([time_id_value], dtype=torch.long, device=device)
            generate_kwargs['time_id'] = current_time_id

        with torch.no_grad():
            tokens = prior.generate(**generate_kwargs).cpu().numpy()

        overlap_len = seq_len // 2
        curr_start_token = torch.from_numpy(tokens[:, -overlap_len:]).to(device)
        token_blocks.append(tokens)

    return token_blocks


def _decode_bottom_blocks(
    vqvae,
    bottom_tokens_list: List[np.ndarray],
    bottom_grid: Optional[list],
    device: torch.device,
) -> List[np.ndarray]:
    reconstructed_spectrograms = []
    for bottom_tokens in bottom_tokens_list:
        bottom_tokens_tensor = torch.from_numpy(bottom_tokens).to(device)
        with torch.no_grad():
            decoded_specs = _decode_bottom_indices(vqvae, bottom_tokens_tensor, bottom_grid, device)
        reconstructed_spectrograms.append(decoded_specs)
    return reconstructed_spectrograms

def _save_decoded_spectrograms(specs: np.ndarray, save_dir: str) -> None:
    spec_dir = os.path.join(save_dir, 'spectrograms')
    os.makedirs(spec_dir, exist_ok=True)

    np.save(os.path.join(spec_dir, 'bottom_decoded_specs.npy'), specs)

    for i in range(specs.shape[0]):
        img = specs[i, :, :, 0]
        plt.figure(figsize=(6, 4))
        plt.imshow(img, origin='lower', aspect='auto')
        plt.colorbar()
        plt.title(f'Decoded Bottom Spectrogram {i}')
        plt.tight_layout()
        plt.savefig(os.path.join(spec_dir, f'bottom_spec_{i:03d}.png'), dpi=150)
        plt.close()

def _decode_bottom_indices(
    vqvae,
    indices: torch.Tensor,
    grid: Optional[list],
    device: torch.device,
) -> np.ndarray:
    if indices.ndim != 2:
        raise ValueError(f'Expected indices shape (B, T), got {tuple(indices.shape)}')
    if not (isinstance(grid, list) and len(grid) == 2):
        raise ValueError('Bottom grid is required to reshape indices into (H, W)')

    # grid[0] is now Time, grid[1] is now Frequency
    time_steps, freq_bins = int(grid[0]), int(grid[1])
    
    if time_steps * freq_bins != indices.shape[1]:
        raise ValueError(f'Grid {grid} does not match seq_len={indices.shape[1]}')

    # View as (Batch, Time, Freq), then transpose to (Batch, Freq, Time)
    idx_2d = indices.view(indices.shape[0], time_steps, freq_bins).transpose(1, 2).contiguous().long().to(device)
    
    vqvae.eval()
    with torch.no_grad():
        emb = vqvae.vq.embedding[idx_2d]  # (B, Freq, Time, D)
        z_q = emb.permute(0, 3, 1, 2).contiguous()  # (B, D, Freq, Time)
        x_hat = vqvae.decoder(z_q)
        if vqvae.activation_layer is not None:
            x_hat = vqvae.activation_layer(x_hat)

    return x_hat.detach().cpu().permute(0, 2, 3, 1).numpy()


def main():
    parser = argparse.ArgumentParser(description='Generate music from hierarchical transformer priors.')
    parser.add_argument('--top_config', type=str, default=None, help='Path to top prior config.yaml or run directory')
    parser.add_argument('--middle_config', type=str, default=None, help='Path to middle prior config.yaml or run directory')
    parser.add_argument('--bottom_config', type=str, default=None, help='Path to bottom prior config.yaml or run directory')
    parser.add_argument('--top_run_root', type=str, default=DEFAULT_TOP_RUN_ROOT, help='Default top run root used when --top_config is not provided')
    parser.add_argument('--middle_run_root', type=str, default=DEFAULT_MIDDLE_RUN_ROOT, help='Default middle run root used when --middle_config is not provided')
    parser.add_argument('--bottom_run_root', type=str, default=DEFAULT_BOTTOM_RUN_ROOT, help='Default bottom run root used when --bottom_config is not provided')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature for all priors')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling (None disables top-k)')
    parser.add_argument('--chunks_to_generate', type=int, default=3, help='Number of chunks to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible generation (set to negative to disable)')
    parser.add_argument('--audio_method', type=str, default='griffinlim', choices=['griffinlim', 'istft'], help='Spectrogram inversion method')
    parser.add_argument('--save_root', type=str, default='samples/transformer_hierarchical_generated', help='Root directory for generated outputs')
    args = parser.parse_args()

    if args.seed is not None and args.seed < 0:
        args.seed = None
    _set_seed(args.seed)

    try:
        save_dir = generate_hierarchical_music(args)
        print(f'Done! Saved generated samples to {save_dir}')
    except Exception as e:
        _debug(f'Generation failed in main with error: {type(e).__name__}: {e}')
        raise


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'Using deterministic seed: {seed}')


def linear_crossfading(spec_chunk_1, spec_chunk_2, overlap_len):
    """
    Linearly crossfade between two spectrogram chunks over the specified overlap length
    Formula: S_blended = (1-alfa) * S_1 + alfa * S_2
    
    @spec_chunk_1: Spectrogram chunk from the first block (shape: [channels, time_frames, freq_bins])
    @spec_chunk_2: Spectrogram chunk from the second block (shape: [channels, time_frames, freq_bins])
    @overlap_len: Number of time frames to use for the crossfade (must be > 0)
    @returns: Blended spectrogram chunk with the same shape as the input chunks
    """
    assert overlap_len > 0, "Overlap length must be greater than 0 for crossfading"

    # Make copies to prevent in-place mutation bugs
    c1 = spec_chunk_1.copy()
    c2 = spec_chunk_2.copy()
    
    fade_in = np.linspace(0, 1, num=overlap_len).reshape(1, 1, overlap_len, 1)
    fade_out = 1 - fade_in

    # Apply crossfade to the overlapping regions in the Time dimension (index 2)
    blended_overlap = (c1[:, :, -overlap_len:] * fade_out) + (c2[:, :, :overlap_len] * fade_in)

    # Concatenate along the Time dimension (axis=2)
    combined = np.concatenate([
        c1[:, :, :-overlap_len], 
        blended_overlap, 
        c2[:, :, overlap_len:]
    ], axis=2)
    return combined


def generate_hierarchical_music(args) -> str:
    _debug('Resolving config paths...')
    top_transformer_prior_config_path = _resolve_prior_config_path(args.top_config, args.top_run_root, 'top')
    middle_transformer_prior_config_path = _resolve_prior_config_path(args.middle_config, args.middle_run_root, 'middle')
    bottom_transformer_prior_config_path = _resolve_prior_config_path(args.bottom_config, args.bottom_run_root, 'bottom')

    print(f'Top config: {top_transformer_prior_config_path}')
    print(f'Middle config: {middle_transformer_prior_config_path}')
    print(f'Bottom config: {bottom_transformer_prior_config_path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _debug(f'Using device: {device}')

    # Load the three trained priors.
    _debug('Loading transformer priors...')
    top_prior, top_config, _ = load_transformer_prior('top', top_transformer_prior_config_path, device)
    middle_prior, middle_config, _ = load_transformer_prior('middle', middle_transformer_prior_config_path, device)
    bottom_prior, bottom_config, _ = load_transformer_prior('bottom', bottom_transformer_prior_config_path, device)

    top_seq_len = int(top_config['model']['inferred_seq_lens']['top'])
    middle_seq_len = int(middle_config['model']['inferred_seq_lens']['middle'])
    bottom_seq_len = int(bottom_config['model']['inferred_seq_lens']['bottom'])
    bottom_grid = bottom_config['model'].get('inferred_grids', {}).get('bottom')

    _debug('Resolving min_max_values.pkl path...')
    min_max_values_path = _resolve_min_max_values_path(bottom_config)

    _debug('Loading bottom VQ-VAE decoder...')
    vqvae_bottom_decoder = load_jukebox_model(
        bottom_config['vqvae']['bottom_model_dir'],
        'bottom',
        device,
        bottom_config['vqvae']['weights_file'],
    )
    vqvae_bottom_decoder.eval()

    # Step 1: Top-Level Unrolling (The Composer)
    # Generate global structure block-by-block, carrying overlap context.
    top_tokens_list = _generate_level_tokens(
        prior=top_prior,
        seq_len=top_seq_len,
        chunks_to_generate=args.chunks_to_generate,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        upper_tokens_list=None,
        use_time_id=True,
    )
    print('Top-level generation complete. Generated tokens for each block have shape:', top_tokens_list[0].shape if top_tokens_list else None)

    # Step 2: Hierarchical Upsampling (The Performers)
    # Middle prior conditions on top tokens, then bottom prior on middle tokens.
    middle_tokens_list = _generate_level_tokens(
        prior=middle_prior,
        seq_len=middle_seq_len,
        chunks_to_generate=args.chunks_to_generate,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        upper_tokens_list=top_tokens_list,
        use_time_id=False,
    )
    print('Middle-level generation complete. Generated tokens for each block have shape:', middle_tokens_list[0].shape if middle_tokens_list else None)

    bottom_tokens_list = _generate_level_tokens(
        prior=bottom_prior,
        seq_len=bottom_seq_len,
        chunks_to_generate=args.chunks_to_generate,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        upper_tokens_list=middle_tokens_list,
        use_time_id=False,
    )
    print('Generation complete. Bottom tokens length:', len(bottom_tokens_list),
          'with each block having shape:', bottom_tokens_list[0].shape if bottom_tokens_list else None)
    print('Decoding bottom tokens into spectrograms...')

    # Step 3: Spectrogram Crossfading (The Audio Engineer)
    # Decode each chunk then crossfade overlaps to remove block boundaries.
    reconstructed_spectrograms = _decode_bottom_blocks(
        vqvae=vqvae_bottom_decoder,
        bottom_tokens_list=bottom_tokens_list,
        bottom_grid=bottom_grid,
        device=device,
    )
    print('Decoded spectrograms for all blocks. Each block has shape:', reconstructed_spectrograms[0].shape if reconstructed_spectrograms else None)

    final_spectrogram = reconstructed_spectrograms[0].copy()
    for i in range(1, len(reconstructed_spectrograms)):
        next_chunk = reconstructed_spectrograms[i].copy()  # (Batch, Freq, Time, Channels)
        overlap_frames = next_chunk.shape[2] // 2
        final_spectrogram = linear_crossfading(final_spectrogram, next_chunk, overlap_frames)
    print('Spectrogram reconstruction and crossfading complete. Final spectrogram shape:', final_spectrogram.shape)

    # Step 4: Spectrogram Inversion (The Mastering Engineer)
    # Convert final spectrograms back to audio and save all artifacts.
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join(args.save_root, current_time)

    _debug(f'Loading min/max values from: {min_max_values_path}')
    with open(min_max_values_path, 'rb') as f:
        min_max_values = pickle.load(f)

    _save_decoded_spectrograms(final_spectrogram, save_dir)
    min_max_list = _prepare_min_max_values(min_max_values, final_spectrogram.shape[0])

    sound_generator = SoundGenerator(vqvae_bottom_decoder, hop_length=HOP_LENGTH)
    audio_signals = sound_generator.convert_spectrograms_to_audio(
        final_spectrogram, min_max_list, method=args.audio_method
    )

    audio_dir = os.path.join(save_dir, 'audio')
    os.makedirs(audio_dir, exist_ok=True)
    for i, signal in enumerate(audio_signals):
        sf.write(os.path.join(audio_dir, f'sample_{i}.wav'), signal, SAMPLE_RATE)

    return save_dir


if __name__ == '__main__':
    main()
