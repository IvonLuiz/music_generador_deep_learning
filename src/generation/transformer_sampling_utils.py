import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from generation.audio_inversion import AudioInversionConfig
from generation.soundgenerator import SoundGenerator
from train_scripts.jukebox_utils import load_jukebox_model
from windowed_data_utils import extract_prefix_from_previous_window, level_grid_info


DEFAULT_TOP_RUN_ROOT = "models/transformer_prior/jukebox_maestro_top_transformer_prior"
DEFAULT_MIDDLE_RUN_ROOT = "models/transformer_prior/jukebox_maestro_middle_transformer_prior"
DEFAULT_BOTTOM_RUN_ROOT = "models/transformer_prior/jukebox_maestro_bottom_transformer_prior"


def resolve_latest_config_path(run_root: str, level_name: str) -> str:
    """!
    @brief Resolve the newest `config.yaml` inside a prior run root.
    @param run_root File, run directory, or directory containing timestamped runs.
    @param level_name Level name used in error messages.
    @return Resolved config path.
    """
    if os.path.isfile(run_root):
        return run_root

    if not os.path.isdir(run_root):
        raise FileNotFoundError(f"{level_name} run root does not exist: {run_root}")

    direct_config = os.path.join(run_root, "config.yaml")
    if os.path.isfile(direct_config):
        return direct_config

    candidates = []
    for entry in os.listdir(run_root):
        entry_path = os.path.join(run_root, entry)
        if not os.path.isdir(entry_path):
            continue
        config_path = os.path.join(entry_path, "config.yaml")
        if os.path.isfile(config_path):
            candidates.append(config_path)

    if not candidates:
        raise FileNotFoundError(f"No config.yaml found in {run_root} for level {level_name}.")

    return max(candidates, key=os.path.getmtime)


def resolve_prior_config_path(
    explicit_path: Optional[str],
    default_run_root: str,
    level_name: str,
) -> str:
    """!
    @brief Resolve a prior config from an explicit path or a default run root.
    @param explicit_path Optional config path, checkpoint path, or run directory.
    @param default_run_root Default root used when no explicit path is provided.
    @param level_name Level name used in error messages.
    @return Resolved config path.
    """
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f"{level_name} config path not found: {explicit_path}")
        if os.path.isdir(explicit_path):
            return resolve_latest_config_path(explicit_path, level_name)
        return explicit_path

    return resolve_latest_config_path(default_run_root, level_name)


def generate_level_windows(
    prior,
    seq_len: int,
    num_samples: int,
    start_frames: List[int],
    device: torch.device,
    temperature: float,
    top_k: Optional[int],
    upper_tokens_list: Optional[List[np.ndarray]] = None,
    second_upper_tokens_list: Optional[List[np.ndarray]] = None,
    timing_list: Optional[List[torch.Tensor]] = None,
    level_name: str = "level",
    progress_interval: int = 0,
    level_time_frames: Optional[int] = None,
    level_grid: Optional[List[int]] = None,
    use_overlap_prefixes: bool = False,
) -> List[np.ndarray]:
    """!
    @brief Autoregressively sample token windows for one hierarchy level.
    @param prior Loaded Transformer prior.
    @param seq_len Number of tokens generated per window.
    @param num_samples Batch size.
    @param start_frames Window start frames in raw spectrogram frame units.
    @param device Torch device.
    @param temperature Sampling temperature.
    @param top_k Optional top-k filtering value.
    @param upper_tokens_list Optional primary conditioning token windows.
    @param second_upper_tokens_list Optional secondary conditioning token windows.
    @param timing_list Optional timing tensors aligned with `start_frames`.
    @param level_name Label used for progress output.
    @param progress_interval Autoregressive progress interval. Use 0 to disable.
    @param level_time_frames Window size in spectrogram frames.
    @param level_grid Latent grid `[time_cols, freq_bins]`.
    @param use_overlap_prefixes Whether to copy overlapping previous tokens as fixed prefixes.
    @return List of generated token arrays shaped `(B, seq_len)`.
    """
    token_blocks = []
    previous_tokens = None
    previous_start_frame = None

    for chunk, start_frame in enumerate(start_frames):
        upper_indices = None
        if upper_tokens_list is not None:
            upper_indices = torch.from_numpy(upper_tokens_list[chunk]).to(device)

        second_upper_indices = None
        if second_upper_tokens_list is not None:
            second_upper_indices = torch.from_numpy(second_upper_tokens_list[chunk]).to(device)

        start_tokens = None
        if use_overlap_prefixes and previous_tokens is not None:
            if level_time_frames is None or level_grid is None:
                raise ValueError("level_time_frames and level_grid are required for overlap prefix sampling.")
            prefix = extract_prefix_from_previous_window(
                previous_tokens=previous_tokens,
                previous_start_frame=previous_start_frame,
                current_start_frame=start_frame,
                level_time_frames=level_time_frames,
                level_grid=level_grid,
            )
            if prefix is not None and prefix.shape[1] > 0:
                start_tokens = torch.from_numpy(prefix).to(device)

        generate_kwargs = {
            "batch_size": int(num_samples),
            "start_tokens": start_tokens,
            "upper_indices": upper_indices,
            "second_upper_indices": second_upper_indices,
            "seq_len": int(seq_len),
            "temperature": float(temperature),
            "top_k": top_k,
            "device": device,
        }
        if progress_interval and progress_interval > 0:
            generate_kwargs["progress_label"] = f"{level_name} window {chunk + 1}/{len(start_frames)}"
            generate_kwargs["progress_interval"] = int(progress_interval)
        if timing_list is not None:
            generate_kwargs["timing"] = timing_list[chunk].to(device=device, dtype=torch.float32)

        with torch.no_grad():
            tokens = prior.generate(**generate_kwargs).cpu().numpy()

        token_blocks.append(tokens)
        previous_tokens = tokens
        previous_start_frame = start_frame

    return token_blocks


def compute_windowed_step(
    level_time_frames: int,
    level_grid: List[int],
    overlap_fraction: float,
) -> Tuple[int, float, int, int]:
    """!
    @brief Convert an overlap fraction into a token-column-aligned frame hop.
    @param level_time_frames Window size in spectrogram frames.
    @param level_grid Latent grid `[time_cols, freq_bins]`.
    @param overlap_fraction Requested overlap fraction in `[0, 1)`.
    @return Tuple `(step_frames, effective_overlap, overlap_cols, hop_cols)`.
    """
    if not 0.0 <= overlap_fraction < 1.0:
        raise ValueError(f"overlap_fraction must be in [0, 1), got {overlap_fraction}")

    time_cols, _, frames_per_token_col = level_grid_info(level_time_frames, level_grid)
    overlap_cols = int(round(time_cols * overlap_fraction))
    overlap_cols = min(max(overlap_cols, 0), time_cols - 1)
    hop_cols = time_cols - overlap_cols
    step_frames = hop_cols * frames_per_token_col
    effective_overlap = overlap_cols / time_cols
    return step_frames, effective_overlap, overlap_cols, hop_cols


def resolve_decode_context_cols(requested_context_cols: int, chunk_time_cols: int) -> int:
    """!
    @brief Resolve automatic timeline-decode context column settings.
    @param requested_context_cols User-requested context columns. Negative means automatic.
    @param chunk_time_cols Native decoder chunk width in latent time columns.
    @return Effective non-negative context column count.
    """
    if requested_context_cols < 0:
        return max(1, int(chunk_time_cols) // 2)
    return int(requested_context_cols)


def resolve_quantized_path(transformer_config: dict) -> Optional[str]:
    """!
    @brief Resolve the quantized-data root from a transformer config.
    @param transformer_config Parsed transformer config.
    @return Expanded quantized-data path, or None.
    """
    dataset_cfg = transformer_config.get("dataset", {}) if isinstance(transformer_config, dict) else {}
    quantized_path = dataset_cfg.get("quantized_data_path")
    if not quantized_path:
        return None
    return os.path.expanduser(str(quantized_path))


def load_windowed_quantization_config(transformer_config: dict) -> Tuple[Dict, Optional[str]]:
    """!
    @brief Load `windowed_quantization_config.json` referenced by a transformer config.
    @param transformer_config Parsed transformer config.
    @return Tuple `(quantization_config, quantized_path)`.
    """
    quantized_path = resolve_quantized_path(transformer_config)
    if not quantized_path:
        return {}, None

    config_path = os.path.join(quantized_path, "windowed_quantization_config.json")
    if not os.path.isfile(config_path):
        print(f"Warning: windowed_quantization_config.json not found at {config_path}; using transformer config fallbacks.")
        return {}, quantized_path

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f), quantized_path


def infer_slice_len(
    model_cfg: dict,
    target_seq_len: int,
    inferred_len_key: str,
    inferred_stride_key: str,
) -> int:
    """!
    @brief Infer a conditioning token slice length from saved model metadata.
    @param model_cfg Model config dictionary.
    @param target_seq_len Target prior sequence length.
    @param inferred_len_key Config key containing an explicit conditioning length.
    @param inferred_stride_key Config key containing an upsample stride.
    @return Conditioning token slice length.
    """
    inferred_len = int(model_cfg.get(inferred_len_key, 0))
    if inferred_len > 0:
        return inferred_len

    inferred_stride_value = model_cfg.get(inferred_stride_key, 0)
    if isinstance(inferred_stride_value, (tuple, list)):
        if len(inferred_stride_value) != 2:
            raise ValueError(f"Expected {inferred_stride_key} to have two values, got {inferred_stride_value}")
        inferred_stride = int(inferred_stride_value[0]) * int(inferred_stride_value[1])
    else:
        inferred_stride = int(inferred_stride_value)

    if inferred_stride > 0:
        if target_seq_len % inferred_stride != 0:
            raise ValueError(
                f"Cannot infer conditioning slice length: target_seq_len={target_seq_len} "
                f"is not divisible by stride={inferred_stride} ({inferred_stride_key})."
            )
        return target_seq_len // inferred_stride

    raise ValueError(f"Missing both {inferred_len_key} and {inferred_stride_key} in saved model config.")


def assemble_spectrogram_chunks(
    spec_chunks: List[np.ndarray],
    start_frames: List[int],
    total_frames: int,
) -> np.ndarray:
    """!
    @brief Place decoded spectrogram chunks into a timeline with linear crossfades.
    @param spec_chunks Decoded chunks shaped `(B, F, T, C)`.
    @param start_frames Chunk start positions in spectrogram frames.
    @param total_frames Final timeline length.
    @return Assembled spectrogram batch.
    """
    if not spec_chunks:
        raise ValueError("No spectrogram chunks to assemble.")
    if len(spec_chunks) == 1:
        return spec_chunks[0][:, :, :total_frames, :].copy()
    if len(spec_chunks) != len(start_frames):
        raise ValueError(f"spec_chunks length ({len(spec_chunks)}) != start_frames length ({len(start_frames)})")

    first = spec_chunks[0]
    batch_size, freq_bins, _, channels = first.shape
    assembled = np.zeros((batch_size, freq_bins, total_frames, channels), dtype=np.float32)
    current_end = 0

    for chunk, start_frame in zip(spec_chunks, start_frames):
        start = int(start_frame)
        if start >= total_frames:
            continue
        usable = min(chunk.shape[2], total_frames - start)
        if usable <= 0:
            continue

        chunk = chunk[:, :, :usable, :].astype(np.float32, copy=False)
        end = start + usable

        if start >= current_end:
            assembled[:, :, start:end, :] = chunk
            current_end = max(current_end, end)
            continue

        overlap_end = min(current_end, end)
        overlap_len = max(0, overlap_end - start)
        if overlap_len > 0:
            fade_in = np.linspace(0.0, 1.0, num=overlap_len, dtype=np.float32).reshape(1, 1, overlap_len, 1)
            fade_out = 1.0 - fade_in
            assembled[:, :, start:overlap_end, :] = (
                assembled[:, :, start:overlap_end, :] * fade_out
                + chunk[:, :, :overlap_len, :] * fade_in
            )

        if end > overlap_end:
            chunk_start = overlap_len
            assembled[:, :, overlap_end:end, :] = chunk[:, :, chunk_start:chunk_start + (end - overlap_end), :]
        current_end = max(current_end, end)

    return assembled


def decode_full_level_spectrogram(
    level: str,
    vqvae_ref: str,
    full_tokens: np.ndarray,
    level_grid: Optional[list],
    total_frames: int,
    device: torch.device,
    weights_file: str,
    save_dir: str,
    context_cols: int = 0,
) -> np.ndarray:
    """!
    @brief Decode a full token timeline through a level-specific Jukebox VQ-VAE.
    @param level Jukebox level name.
    @param vqvae_ref VQ-VAE run/config/checkpoint reference.
    @param full_tokens Full flattened token timeline.
    @param level_grid Native level grid `[time_cols, freq_bins]`.
    @param total_frames Final spectrogram frame count.
    @param device Torch device.
    @param weights_file VQ-VAE checkpoint filename.
    @param save_dir Output run directory.
    @param context_cols Extra latent context columns for chunked timeline decoding.
    @return Decoded spectrogram batch shaped `(B, F, T, 1)`.
    """
    if not (isinstance(level_grid, (list, tuple)) and len(level_grid) == 2):
        raise ValueError(f"{level} grid is required to decode full generated indices.")

    freq_bins = int(level_grid[1])
    if full_tokens.shape[1] % freq_bins != 0:
        raise ValueError(f"{level} full token length {full_tokens.shape[1]} is not divisible by freq_bins={freq_bins}.")

    total_token_cols = full_tokens.shape[1] // freq_bins
    dynamic_grid = [total_token_cols, freq_bins]
    chunk_cols = int(level_grid[0])
    if chunk_cols <= 0:
        raise ValueError(f"{level} grid has invalid time columns: {level_grid}")

    context_cols = max(0, int(context_cols))
    context_suffix = f" plus {context_cols} context columns" if context_cols else ""
    print(
        f"Decoding full {level} token timeline with dynamic grid {dynamic_grid} "
        f"in chunks of {chunk_cols} token columns{context_suffix}..."
    )

    from generation.transformer_io_utils import decode_jukebox_token_timeline, save_level_spectrograms

    vqvae = load_jukebox_model(vqvae_ref, level, device, weights_file)
    tokens_tensor = torch.from_numpy(full_tokens.astype(np.int64, copy=False)).to(device)
    decoded_specs = decode_jukebox_token_timeline(
        vqvae=vqvae,
        tokens=tokens_tensor,
        grid=dynamic_grid,
        device=device,
        chunk_time_cols=chunk_cols,
        context_cols=context_cols,
        trim_frames=total_frames,
    )
    spectrogram_dir = save_level_spectrograms(
        decoded_specs,
        save_dir,
        level,
        root_subdir="spectrograms",
        include_level_subdir=False,
        npy_subdir="npy",
        npy_filename=f"{level}_full_decoded_specs.npy",
        filename_prefix=f"{level}_full_spectrogram",
        title_template=f"{level.capitalize()} full generated spectrogram {{index}}",
        cmap="magma",
        figsize=(12, 4),
    )
    print(f"Saved full {level} spectrograms to {spectrogram_dir}")

    del vqvae, tokens_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return decoded_specs


def decode_token_blocks(
    vqvae,
    tokens_list: List[np.ndarray],
    level_grid: Optional[list],
    device: torch.device,
) -> List[np.ndarray]:
    """!
    @brief Decode independent token blocks through an already-loaded VQ-VAE.
    @param vqvae Loaded Jukebox VQ-VAE decoder.
    @param tokens_list Token blocks shaped `(B, seq_len)`.
    @param level_grid Native latent grid.
    @param device Torch device.
    @return List of decoded spectrogram chunks.
    """
    from generation.transformer_io_utils import decode_jukebox_indices

    decoded_chunks = []
    for tokens in tokens_list:
        tokens_tensor = torch.from_numpy(tokens).to(device)
        with torch.no_grad():
            decoded_specs = decode_jukebox_indices(vqvae, tokens_tensor, level_grid, device)
        decoded_chunks.append(decoded_specs)
    return decoded_chunks


def fixed_db_min_max_values(count: int, min_db: float, max_db: float) -> List[Dict[str, float]]:
    """!
    @brief Build fixed dB denormalization metadata.
    @param count Number of spectrograms in the batch.
    @param min_db dB value represented by normalized 0.
    @param max_db dB value represented by normalized 1.
    @return List of min/max dictionaries.
    """
    if max_db <= min_db:
        raise ValueError(f"fixed max dB must be > min dB, got min={min_db}, max={max_db}")
    return [{"min": float(min_db), "max": float(max_db)} for _ in range(count)]


def _coerce_audio_config(
    audio_config: Optional[AudioInversionConfig],
    audio_method: Optional[str],
    gradient_steps: int,
    gradient_lr: float,
    gradient_chunk_frames: int,
    gradient_overlap_frames: int,
    decorsiere_alpha: float,
    decorsiere_lr: float,
    decorsiere_history_size: int,
) -> AudioInversionConfig:
    if audio_config is not None:
        return audio_config
    return AudioInversionConfig(
        method=audio_method or "gradient",
        gradient_steps=gradient_steps,
        gradient_lr=gradient_lr,
        gradient_chunk_frames=gradient_chunk_frames,
        gradient_overlap_frames=gradient_overlap_frames,
        decorsiere_alpha=decorsiere_alpha,
        decorsiere_lr=decorsiere_lr,
        decorsiere_history_size=decorsiere_history_size,
    )


def save_audio_from_spectrogram(
    spectrograms: np.ndarray,
    min_max_values,
    save_dir: str,
    filename: str,
    hop_length: int,
    sample_rate: int,
    frame_size: int,
    spectrogram_type: str,
    n_mels: int,
    audio_config: Optional[AudioInversionConfig] = None,
    audio_method: Optional[str] = None,
    gradient_steps: int = 1024,
    gradient_lr: float = 0.0005,
    gradient_chunk_frames: int = 8192,
    gradient_overlap_frames: int = 2048,
    decorsiere_alpha: float = 0.3,
    decorsiere_lr: float = 1.0,
    decorsiere_history_size: int = 10,
) -> str:
    """!
    @brief Convert decoded normalized spectrograms to audio and save the first sample.
    @param spectrograms Decoded spectrogram batch shaped `(B, F, T, 1)`.
    @param min_max_values Dataset min/max metadata or None for fixed dB scale.
    @param save_dir Generation output directory.
    @param filename Output filename inside the `audio` directory.
    @param hop_length STFT/mel hop length.
    @param sample_rate Audio sample rate.
    @param frame_size FFT frame size.
    @param spectrogram_type Spectrogram type: `linear` or `mel`.
    @param n_mels Number of mel bins when using mel spectrograms.
    @param audio_config Optional structured inversion config.
    @return Full path of the written waveform file.
    """
    resolved_audio_config = _coerce_audio_config(
        audio_config,
        audio_method,
        gradient_steps,
        gradient_lr,
        gradient_chunk_frames,
        gradient_overlap_frames,
        decorsiere_alpha,
        decorsiere_lr,
        decorsiere_history_size,
    )
    if resolved_audio_config.use_fixed_db_scale or min_max_values is None:
        min_max_list = fixed_db_min_max_values(
            spectrograms.shape[0],
            resolved_audio_config.fixed_min_db,
            resolved_audio_config.fixed_max_db,
        )
    else:
        from generation.transformer_io_utils import prepare_min_max_values

        min_max_list = prepare_min_max_values(min_max_values, spectrograms.shape[0])

    sound_generator = SoundGenerator(
        None,
        hop_length=hop_length,
        sample_rate=sample_rate,
        n_fft=frame_size,
        spectrogram_type=spectrogram_type,
        n_mels=n_mels,
    )
    audio_signals = sound_generator.convert_spectrograms_to_audio(
        spectrograms,
        min_max_list,
        inversion_config=resolved_audio_config,
    )

    audio_dir = os.path.join(save_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, filename)
    sf.write(audio_path, audio_signals[0], sample_rate)
    return audio_path
