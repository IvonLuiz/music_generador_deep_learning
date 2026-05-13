import math
import os
from typing import List, Optional, Tuple

import numpy as np
import torch


def source_stem(file_path: str) -> str:
    """!
    @brief Strip the `.npy` suffix and return the filename stem.
    @param file_path Path to the source file.
    @return Filename stem without the `.npy` extension.
    """
    return os.path.basename(file_path).replace('.npy', '')


def build_level_starts(total_frames: int, window_size: int, step: int) -> List[int]:
    """!
    @brief Build window start indices for a given level, ensuring coverage of the entire source even if it means a shorter final window.
    @param total_frames The total number of frames in the source audio.
    @param window_size The size of the window in frames for the level.
    @param step The step size in frames between the starts of consecutive windows.
    @return A list of start frame indices for the windows. If total_frames is less than or equal to window_size, returns a single start index of 0 to cover the entire source with a single window.
    """
    if window_size <= 0:
        raise ValueError(f'window_size must be > 0, got {window_size}')
    if step <= 0:
        raise ValueError(f'step must be > 0, got {step}')
    if total_frames <= window_size:
        return [0]

    starts = list(range(0, total_frames - window_size + 1, step))
    last_start = total_frames - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def build_windowed_starts(total_frames: int, window_size: int, step: int) -> List[int]:
    """!
    @brief Build fixed-hop window starts for continuation generation.
    @param total_frames Total number of frames in the source audio.
    @param window_size Window size in frames.
    @param step Step size in frames between windows.
    @return List of start frame indices. The final window may extend past the end and is trimmed later.
    """
    if window_size <= 0:
        raise ValueError(f'window_size must be > 0, got {window_size}')
    if step <= 0:
        raise ValueError(f'step must be > 0, got {step}')

    starts = [0]
    while starts[-1] + window_size < total_frames:
        starts.append(starts[-1] + step)
    return starts


def extract_window(spec: np.ndarray, start_frame: int, window_size: int) -> np.ndarray:
    """!
    @brief Extract a spectrogram window and pad if it is shorter than the target size.
    @param spec Spectrogram array shaped `(F, T)`.
    @param start_frame Start frame index for the window.
    @param window_size Window length in frames.
    @return Windowed spectrogram chunk shaped `(F, window_size)`.
    """
    chunk = spec[:, start_frame:start_frame + window_size]
    if chunk.shape[1] < window_size:
        pad_width = window_size - chunk.shape[1]
        chunk = np.pad(chunk, ((0, 0), (0, pad_width)), mode='constant')
    return chunk.astype(np.float32, copy=False)


def build_timing_tensor(
    start_frame: int,
    total_frames: int,
    sample_rate: int,
    hop_length: int,
) -> torch.Tensor:
    """!
    @brief Build a timing tensor describing the window position in the full source.
    @param start_frame Start frame index for the window.
    @param total_frames Total number of frames in the full source.
    @param sample_rate Audio sample rate in Hz.
    @param hop_length Hop length used to compute frame times.
    @return Tensor `[start_time_s, total_duration_s, fraction_elapsed]`.
    """
    start_time_s = (start_frame * hop_length) / sample_rate
    total_duration_s = (total_frames * hop_length) / sample_rate
    fraction = start_time_s / max(total_duration_s, 1e-6)
    return torch.tensor([start_time_s, total_duration_s, fraction], dtype=torch.float32)


def build_timing_schedule(
    start_frames: List[int],
    hop_length: int,
    sample_rate: int,
    total_source_frames: int,
) -> List[torch.Tensor]:
    """
    Build timing metadata exactly like windowed quantization preprocessing.

    Training stores [start_time_s, full_source_duration_s, fraction_elapsed],
    where full_source_duration_s comes from the original song length, not from
    the short generated clip length.
    """
    schedule = []
    for start_frame in start_frames:
        timing_tensor = build_timing_tensor(
            start_frame,
            total_source_frames,
            sample_rate,
            hop_length
        )
        schedule.append(timing_tensor.unsqueeze(0))
    return schedule


def build_timing_schedule_with_offset(
    start_frames: List[int],
    hop_length: int,
    sample_rate: int,
    total_source_frames: int,
    base_start_frame: int = 0,
) -> List[torch.Tensor]:
    schedule = []
    for start_frame in start_frames:
        timing_tensor = build_timing_tensor(
            int(base_start_frame) + int(start_frame),
            total_source_frames,
            sample_rate,
            hop_length,
        )
        schedule.append(timing_tensor.unsqueeze(0))
    return schedule


def seconds_to_frames(seconds: float, sample_rate: int, hop_length: int) -> int:
    return max(1, int(round(float(seconds) * int(sample_rate) / int(hop_length))))


def level_grid_info(level_time_frames: int, level_grid: List[int]) -> Tuple[int, int, int]:
    if not (isinstance(level_grid, (list, tuple)) and len(level_grid) == 2):
        raise ValueError(f'Expected level_grid=[time_cols, freq_bins], got {level_grid}')
    time_cols, freq_bins = int(level_grid[0]), int(level_grid[1])
    if time_cols <= 0 or freq_bins <= 0:
        raise ValueError(f'Invalid level grid: {level_grid}')
    if level_time_frames % time_cols != 0:
        raise ValueError(
            f'level_time_frames={level_time_frames} is not divisible by time_cols={time_cols}'
        )
    return time_cols, freq_bins, level_time_frames // time_cols


def assemble_token_timeline(
    tokens_list: List[np.ndarray],
    start_frames: List[int],
    level_time_frames: int,
    level_grid: List[int],
    total_frames: int,
) -> np.ndarray:
    if not tokens_list:
        return np.array([])
    if len(tokens_list) != len(start_frames):
        raise ValueError(f'tokens_list length ({len(tokens_list)}) != start_frames length ({len(start_frames)})')

    time_cols, freq_bins, frames_per_token_col = level_grid_info(level_time_frames, level_grid)
    total_time_cols = max(time_cols, math.ceil(total_frames / frames_per_token_col))

    batch_size = tokens_list[0].shape[0]
    timeline = np.zeros((batch_size, total_time_cols, freq_bins), dtype=tokens_list[0].dtype)

    for tokens, start_frame in zip(tokens_list, start_frames):
        block = tokens.reshape(batch_size, time_cols, freq_bins)
        start_col = int(round(int(start_frame) / frames_per_token_col))
        if start_col >= total_time_cols:
            continue
        block_end = min(time_cols, total_time_cols - start_col)
        timeline[:, start_col:start_col + block_end, :] = block[:, :block_end, :]

    return timeline.reshape(batch_size, total_time_cols * freq_bins)


def fixed_subwindow_starts(parent_time_frames: int, child_time_frames: int, child_name: str) -> List[int]:
    if parent_time_frames <= 0 or child_time_frames <= 0:
        raise ValueError(
            f'Invalid hierarchy frame sizes: parent={parent_time_frames}, {child_name}={child_time_frames}'
        )
    if parent_time_frames % child_time_frames != 0:
        raise ValueError(
            f'Cannot split one top sequence of {parent_time_frames} frames into exact '
            f'{child_name} windows of {child_time_frames} frames.'
        )
    return list(range(0, parent_time_frames, child_time_frames))


def overlapped_subwindow_starts(
    parent_time_frames: int,
    child_time_frames: int,
    overlap_fraction: float,
    child_name: str,
    align_frames: int = 1,
) -> List[int]:
    if parent_time_frames <= 0 or child_time_frames <= 0:
        raise ValueError(
            f'Invalid hierarchy frame sizes: parent={parent_time_frames}, {child_name}={child_time_frames}'
        )
    if child_time_frames > parent_time_frames:
        raise ValueError(
            f'{child_name} window ({child_time_frames} frames) cannot exceed parent window '
            f'({parent_time_frames} frames).'
        )
    if overlap_fraction < 0.0 or overlap_fraction >= 1.0:
        raise ValueError(f'overlap_fraction must be in [0, 1), got {overlap_fraction}')

    if overlap_fraction == 0.0:
        return fixed_subwindow_starts(parent_time_frames, child_time_frames, child_name)

    hop_frames = max(1, int(round(child_time_frames * (1.0 - overlap_fraction))))
    align_frames = max(1, int(align_frames))
    if align_frames > 1:
        hop_frames = max(align_frames, int(round(hop_frames / align_frames)) * align_frames)
    starts = list(range(0, max(1, parent_time_frames - child_time_frames + 1), hop_frames))
    final_start = parent_time_frames - child_time_frames
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def extract_prefix_from_previous_window(
    previous_tokens: np.ndarray,
    previous_start_frame: int,
    current_start_frame: int,
    level_time_frames: int,
    level_grid: List[int],
) -> Optional[np.ndarray]:
    time_cols, freq_bins, frames_per_token_col = level_grid_info(level_time_frames, level_grid)
    delta_frames = int(current_start_frame) - int(previous_start_frame)
    if delta_frames <= 0:
        return None

    delta_cols = int(round(delta_frames / frames_per_token_col))
    if delta_cols <= 0 or delta_cols >= time_cols:
        return None

    block = previous_tokens.reshape(previous_tokens.shape[0], time_cols, freq_bins)
    return block[:, delta_cols:, :].reshape(previous_tokens.shape[0], -1)


def validate_window_prefixes(
    tokens_list: List[np.ndarray],
    start_frames: List[int],
    level_time_frames: int,
    level_grid: List[int],
    level_name: str,
) -> None:
    if len(tokens_list) <= 1:
        return
    for idx in range(1, len(tokens_list)):
        expected = extract_prefix_from_previous_window(
            previous_tokens=tokens_list[idx - 1],
            previous_start_frame=start_frames[idx - 1],
            current_start_frame=start_frames[idx],
            level_time_frames=level_time_frames,
            level_grid=level_grid,
        )
        if expected is None or expected.shape[1] == 0:
            continue
        actual = tokens_list[idx][:, :expected.shape[1]]
        if not np.array_equal(actual, expected):
            raise RuntimeError(
                f'{level_name} overlap prefix mismatch at window {idx + 1}: '
                f'expected prefix shape {expected.shape}, got {actual.shape}'
            )
    print(f'{level_name} overlap prefix check: OK')


def dynamic_grid_for_tokens(tokens: torch.Tensor, level_grid: Optional[list]) -> Optional[list]:
    if not (isinstance(level_grid, list) and len(level_grid) == 2):
        return level_grid
    freq_bins = int(level_grid[1])
    if freq_bins <= 0 or tokens.shape[1] % freq_bins != 0:
        return level_grid
    return [int(tokens.shape[1] // freq_bins), freq_bins]


def get_token_slice_for_frame(
    full_tokens: np.ndarray,
    start_frame: int,
    level_time_frames: int,
    level_grid: List[int],
    slice_len: int,
    base_start_frame: int = 0,
) -> np.ndarray:
    _, freq_bins, frames_per_token_col = level_grid_info(level_time_frames, level_grid)
    relative_frame = int(start_frame) - int(base_start_frame)
    if relative_frame < 0:
        raise ValueError(
            f'Requested conditioning slice start_frame={start_frame} is before '
            f'base_start_frame={base_start_frame}.'
        )
    token_col_start = int(round(relative_frame / frames_per_token_col))
    start = token_col_start * freq_bins
    end = start + slice_len

    if start >= full_tokens.shape[1]:
        print(
            f'Warning: Requested conditioning slice start_frame={start_frame} is out of bounds '
            f'(token_start={start}, total_len={full_tokens.shape[1]}). Returning zeros.'
        )
        return np.zeros((full_tokens.shape[0], slice_len), dtype=full_tokens.dtype)

    actual_slice = full_tokens[:, start:end]
    if actual_slice.shape[1] < slice_len:
        pad_width = slice_len - actual_slice.shape[1]
        actual_slice = np.pad(actual_slice, ((0, 0), (0, pad_width)), mode='constant')
    return actual_slice
