import os
from typing import List

import numpy as np
import torch


def source_stem(file_path: str) -> str:
    return os.path.basename(file_path).replace('.npy', '')


def build_level_starts(total_frames: int, window_size: int, step: int) -> List[int]:
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
    """Build fixed-hop continuation starts; the final generated window is trimmed during assembly."""
    if window_size <= 0:
        raise ValueError(f'window_size must be > 0, got {window_size}')
    if step <= 0:
        raise ValueError(f'step must be > 0, got {step}')

    starts = [0]
    while starts[-1] + window_size < total_frames:
        starts.append(starts[-1] + step)
    return starts


def extract_window(spec: np.ndarray, start_frame: int, window_size: int) -> np.ndarray:
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
