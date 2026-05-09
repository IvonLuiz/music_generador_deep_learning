import argparse
import glob
import os
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from test_scripts.test_transformer_prior import (
    _assemble_token_timeline,
    _decode_level_to_audio,
    _dynamic_grid_for_tokens,
    _generate_level_windows,
    _validate_window_prefixes,
    load_transformer_prior,
    set_reproducible_seed,
)
from windowed_data_utils import build_timing_tensor


def _pick(root, hint=None):
    fs = sorted(glob.glob(os.path.join(os.path.abspath(os.path.expanduser(root)), '**', '*_window_quantized.pt'), recursive=True))
    if not fs:
        raise FileNotFoundError(f'No *_window_quantized.pt files under {root}')
    if not hint:
        return fs[0]
    hs = [f for f in fs if hint in os.path.basename(f) or hint in f]
    if not hs:
        raise FileNotFoundError(f'No quantized file matching {hint!r} under {root}')
    return hs[0]


def _repeat_tokens(x, n_samples):
    return torch.as_tensor(x, dtype=torch.long).reshape(1, -1).repeat(int(n_samples), 1)


def _timing_from_payload(payload, n_samples):
    timing = payload.get('timing')
    if timing is None:
        timing = build_timing_tensor(
            int(payload.get('start_frame', 0)),
            int(payload.get('total_frames', 2048)),
            22050,
            256,
        )
    return timing.reshape(1, 3).repeat(int(n_samples), 1)


def _source_files(root, source_stem):
    pattern = os.path.join(os.path.abspath(os.path.expanduser(root)), '**', f'{source_stem}__start_*_window_quantized.pt')
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f'No windowed quantized files found for source_stem={source_stem!r}')
    return files


def _save_window_npz(path, arrays):
    np.savez(path, **{f'window_{i:04d}': arr for i, arr in enumerate(arrays)})


def main():
    p = argparse.ArgumentParser(description='Bottom prior test with real conditioning.')
    p.add_argument('--bottom_prior', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--file', default=None)
    p.add_argument('--bottom_vqvae', default=None)
    p.add_argument('--weights_file', default='best_model.pth')
    p.add_argument('--n_samples', type=int, default=1)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top_k', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_root', default='./samples/bottom_prior_conditioned')
    p.add_argument('--audio_method', default='griffinlim')
    p.add_argument('--full_length', action='store_true')
    p.add_argument('--progress_interval', type=int, default=128)
    a = p.parse_args()

    set_reproducible_seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prior, cfg, _ = load_transformer_prior('bottom', a.bottom_prior, device, a.weights_file)
    qfile = _pick(a.data_root, a.file)
    payload = torch.load(qfile, map_location='cpu', weights_only=False)
    if payload.get('format') != 'windowed_v1':
        raise ValueError('Use *_window_quantized.pt files')

    out = os.path.join(os.path.abspath(os.path.expanduser(a.output_root)), datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(out, exist_ok=True)
    vq = a.bottom_vqvae or cfg.get('vqvae', {}).get('bottom_model_dir')
    weights = cfg.get('vqvae', {}).get('weights_file', 'best_model.pth')
    grid = cfg['model'].get('inferred_grids', {}).get('bottom')

    if not a.full_length:
        top = _repeat_tokens(payload['top'], a.n_samples).to(device)
        middle = _repeat_tokens(payload['middle'], a.n_samples).to(device)
        real_bottom = _repeat_tokens(payload['bottom'], a.n_samples)
        timing = _timing_from_payload(payload, a.n_samples).to(device=device, dtype=torch.float32)
        print(f'Conditioning on {qfile}')
        with torch.no_grad():
            generated = prior.generate(
                batch_size=int(a.n_samples),
                start_tokens=None,
                upper_indices=middle,
                second_upper_indices=top if getattr(prior, 'second_conditioner', None) is not None else None,
                seq_len=int(cfg['model']['inferred_seq_lens']['bottom']),
                temperature=float(a.temperature),
                top_k=a.top_k if a.top_k and a.top_k > 0 else None,
                device=device,
                timing=timing,
                progress_label='bottom',
                progress_interval=int(a.progress_interval),
            ).cpu()
        for name, arr in [
            ('generated_bottom', generated.numpy().astype(np.int64)),
            ('real_bottom', real_bottom.numpy().astype(np.int64)),
            ('real_middle', middle.cpu().numpy().astype(np.int64)),
            ('real_top', top.cpu().numpy().astype(np.int64)),
        ]:
            np.save(os.path.join(out, f'{name}.npy'), arr)
        if vq:
            _decode_level_to_audio('bottom', generated, grid, vq, weights, a.audio_method, out, device)
        print(f'Saved bottom-only test outputs to {out}')
        return

    files = _source_files(a.data_root, payload['source_stem'])
    records = []
    for path in files:
        item = torch.load(path, map_location='cpu', weights_only=False)
        if item.get('format') != 'windowed_v1':
            continue
        if 'bottom' not in item.get('eligible_levels', []):
            continue
        records.append((int(item.get('start_frame', 0)), path, item))
    if not records:
        raise ValueError(f'No bottom-eligible windowed records found for {payload["source_stem"]}')
    records.sort(key=lambda x: x[0])

    level_tf = int(cfg.get('dataset', {}).get('level_target_time_frames', {}).get('bottom', 128))
    total_frames = int(records[0][2].get('total_frames', level_tf))
    start_frames = [r[0] for r in records]
    top_list = [_repeat_tokens(r[2]['top'], a.n_samples).numpy().astype(np.int64) for r in records]
    middle_list = [_repeat_tokens(r[2]['middle'], a.n_samples).numpy().astype(np.int64) for r in records]
    real_bottom_list = [_repeat_tokens(r[2]['bottom'], a.n_samples).numpy().astype(np.int64) for r in records]
    timing_list = [_timing_from_payload(r[2], a.n_samples) for r in records]

    print(f'Full-length conditioning on {payload["source_stem"]} with {len(records)} bottom windows')
    generated_list = _generate_level_windows(
        prior=prior,
        seq_len=int(cfg['model']['inferred_seq_lens']['bottom']),
        num_samples=int(a.n_samples),
        start_frames=start_frames,
        device=device,
        temperature=float(a.temperature),
        top_k=a.top_k if a.top_k and a.top_k > 0 else None,
        upper_tokens_list=middle_list,
        second_upper_tokens_list=top_list if getattr(prior, 'second_conditioner', None) is not None else None,
        timing_list=timing_list,
        level_name='bottom',
        progress_interval=int(a.progress_interval),
        level_time_frames=level_tf,
        level_grid=grid,
        use_overlap_prefixes=True,
    )
    _validate_window_prefixes(generated_list, start_frames, level_tf, grid, 'bottom')

    full_generated = _assemble_token_timeline(generated_list, start_frames, level_tf, grid, total_frames).astype(np.int64)
    full_real_bottom = _assemble_token_timeline(real_bottom_list, start_frames, level_tf, grid, total_frames).astype(np.int64)
    np.save(os.path.join(out, 'generated_bottom_full.npy'), full_generated)
    np.save(os.path.join(out, 'real_bottom_full.npy'), full_real_bottom)
    np.save(os.path.join(out, 'start_frames.npy'), np.asarray(start_frames, dtype=np.int64))
    _save_window_npz(os.path.join(out, 'generated_bottom_windows.npz'), [x.astype(np.int64) for x in generated_list])
    _save_window_npz(os.path.join(out, 'real_bottom_windows.npz'), real_bottom_list)

    if vq:
        full_tensor = torch.from_numpy(full_generated)
        full_grid = _dynamic_grid_for_tokens(full_tensor, grid)
        _decode_level_to_audio(
            'bottom',
            full_tensor,
            full_grid,
            vq,
            weights,
            a.audio_method,
            out,
            device,
            chunk_time_cols=int(grid[0]) if isinstance(grid, list) and len(grid) == 2 else None,
            decode_context_cols=max(1, int(grid[0]) // 2) if isinstance(grid, list) and len(grid) == 2 else 0,
            trim_frames=total_frames,
        )
    print(f'Saved full-length bottom-only outputs to {out}')


if __name__ == '__main__':
    main()
