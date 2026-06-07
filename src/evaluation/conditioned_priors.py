from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import torch

from evaluation.core import EvaluationResult
from evaluation.transformer_prior import (
    TransformerDecodeTarget,
    TransformerPriorLoader,
    TransformerTokenDecoder,
    TransformerTokenSampler,
)
from generation.audio_inversion import AudioInversionConfig
from utils import set_global_seed
from windowed_data_utils import (
    assemble_token_timeline,
    build_timing_tensor,
    dynamic_grid_for_tokens,
    validate_window_prefixes,
)


@dataclass
class BottomConditionedPriorConfig:
    """!
    @brief Settings for bottom-prior evaluation with real top/middle conditioning.
    """

    bottom_prior: str
    data_root: str
    file: Optional[str] = None
    bottom_vqvae: Optional[str] = None
    weights_file: str = "best_model.pth"
    n_samples: int = 1
    temperature: float = 1.0
    top_k: Optional[int] = None
    seed: int = 42
    output_root: str = "./samples/bottom_prior_conditioned"
    full_length: bool = False
    progress_interval: int = 128


@dataclass
class MiddleBottomConditionedPriorConfig:
    """!
    @brief Settings for real-top -> generated-middle -> generated-bottom ablation.
    """

    middle_prior: str
    bottom_prior: str
    data_root: str
    file: Optional[str] = None
    bottom_vqvae: Optional[str] = None
    weights_file: str = "best_model.pth"
    n_samples: int = 1
    temperature: float = 1.0
    top_k: Optional[int] = None
    seed: int = 42
    output_root: str = "./samples/middle_bottom_prior_conditioned"
    full_length: bool = False
    progress_interval: int = 128


def _pick_quantized_file(root: str, hint: Optional[str] = None) -> str:
    """!
    @brief Pick one windowed quantized payload from a directory.
    @param root Search root.
    @param hint Optional filename/path substring.
    @return Matching payload path.
    """
    files = sorted(glob.glob(os.path.join(os.path.abspath(os.path.expanduser(root)), "**", "*_window_quantized.pt"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No *_window_quantized.pt files under {root}")
    if not hint:
        return files[0]
    matches = [path for path in files if hint in os.path.basename(path) or hint in path]
    if not matches:
        raise FileNotFoundError(f"No quantized file matching {hint!r} under {root}")
    return matches[0]


def _repeat_tokens(x, n_samples: int) -> torch.Tensor:
    """!
    @brief Repeat one token payload across a requested batch size.
    @param x Token array/tensor.
    @param n_samples Batch size.
    @return Tensor shaped `(n_samples, tokens)`.
    """
    return torch.as_tensor(x, dtype=torch.long).reshape(1, -1).repeat(int(n_samples), 1)


def _timing_from_payload(payload: dict, n_samples: int) -> torch.Tensor:
    """!
    @brief Build/repeat timing metadata from a windowed quantized payload.
    @param payload Windowed quantization payload.
    @param n_samples Batch size.
    @return Timing tensor shaped `(n_samples, 3)`.
    """
    timing = payload.get("timing")
    if timing is None:
        timing = build_timing_tensor(
            int(payload.get("start_frame", 0)),
            int(payload.get("total_frames", 2048)),
            22050,
            256,
        )
    return timing.reshape(1, 3).repeat(int(n_samples), 1)


def _source_files(root: str, source_stem: str):
    """!
    @brief Find all windowed quantized payloads belonging to the same source.
    @param root Search root.
    @param source_stem Source stem stored by quantization preprocessing.
    @return Sorted payload paths.
    """
    pattern = os.path.join(os.path.abspath(os.path.expanduser(root)), "**", f"{source_stem}__start_*_window_quantized.pt")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No windowed quantized files found for source_stem={source_stem!r}")
    return files


def _save_window_npz(path: str, arrays) -> None:
    """!
    @brief Save generated/real token windows using the legacy window key naming.
    @param path Output `.npz` path.
    @param arrays Window arrays.
    """
    np.savez(path, **{f"window_{i:04d}": arr for i, arr in enumerate(arrays)})


class BottomConditionedPriorEvaluator:
    """!
    @brief Samples only the bottom prior while conditioning on real top/middle tokens.
    """

    def __init__(self, config: BottomConditionedPriorConfig, audio_config: Optional[AudioInversionConfig] = None):
        """!
        @brief Initialize evaluator.
        @param config Evaluation settings.
        @param audio_config Audio inversion settings.
        """
        self.config = config
        self.audio_config = audio_config or AudioInversionConfig(method="gradient", use_fixed_db_scale=True)

    def run(self) -> EvaluationResult:
        """!
        @brief Run bottom-conditioned prior evaluation.
        @return Evaluation artifact paths.
        """
        cfg = self.config
        set_global_seed(cfg.seed, deterministic=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prior, prior_cfg, _ = TransformerPriorLoader.load_prior("bottom", cfg.bottom_prior, device, cfg.weights_file)
        qfile = _pick_quantized_file(cfg.data_root, cfg.file)
        payload = torch.load(qfile, map_location="cpu", weights_only=False)
        if payload.get("format") != "windowed_v1":
            raise ValueError("Use *_window_quantized.pt files")

        out = os.path.join(os.path.abspath(os.path.expanduser(cfg.output_root)), datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(out, exist_ok=True)
        vqvae_path = cfg.bottom_vqvae or prior_cfg.get("vqvae", {}).get("bottom_model_dir")
        weights = prior_cfg.get("vqvae", {}).get("weights_file", "best_model.pth")
        grid = prior_cfg["model"].get("inferred_grids", {}).get("bottom")

        if not cfg.full_length:
            return self._run_single_window(prior, prior_cfg, payload, qfile, vqvae_path, weights, grid, out, device)
        return self._run_full_length(prior, prior_cfg, payload, vqvae_path, weights, grid, out, device)

    def _run_single_window(self, prior, prior_cfg, payload, qfile, vqvae_path, weights, grid, out, device) -> EvaluationResult:
        cfg = self.config
        top = _repeat_tokens(payload["top"], cfg.n_samples).to(device)
        middle = _repeat_tokens(payload["middle"], cfg.n_samples).to(device)
        real_bottom = _repeat_tokens(payload["bottom"], cfg.n_samples)
        timing = _timing_from_payload(payload, cfg.n_samples).to(device=device, dtype=torch.float32)
        print(f"Conditioning on {qfile}")
        with torch.no_grad():
            generated = prior.generate(
                batch_size=int(cfg.n_samples),
                start_tokens=None,
                upper_indices=middle,
                second_upper_indices=top if getattr(prior, "second_conditioner", None) is not None else None,
                seq_len=int(prior_cfg["model"]["inferred_seq_lens"]["bottom"]),
                temperature=float(cfg.temperature),
                top_k=cfg.top_k if cfg.top_k and cfg.top_k > 0 else None,
                device=device,
                timing=timing,
                progress_label="bottom",
                progress_interval=int(cfg.progress_interval),
            ).cpu()
        for name, arr in [
            ("generated_bottom", generated.numpy().astype(np.int64)),
            ("real_bottom", real_bottom.numpy().astype(np.int64)),
            ("real_middle", middle.cpu().numpy().astype(np.int64)),
            ("real_top", top.cpu().numpy().astype(np.int64)),
        ]:
            np.save(os.path.join(out, f"{name}.npy"), arr)
        if vqvae_path:
            TransformerTokenDecoder.decode(
                "bottom",
                TransformerDecodeTarget(tokens=generated, grid=grid),
                vqvae_path,
                weights,
                self.audio_config,
                out,
                device,
            )
        print(f"Saved bottom-only test outputs to {out}")
        return EvaluationResult(output_dir=out)

    def _run_full_length(self, prior, prior_cfg, payload, vqvae_path, weights, grid, out, device) -> EvaluationResult:
        cfg = self.config
        records = _bottom_eligible_records(cfg.data_root, payload["source_stem"])
        level_tf = int(prior_cfg.get("dataset", {}).get("level_target_time_frames", {}).get("bottom", 128))
        total_frames = int(records[0][2].get("total_frames", level_tf))
        start_frames = [r[0] for r in records]
        top_list = [_repeat_tokens(r[2]["top"], cfg.n_samples).numpy().astype(np.int64) for r in records]
        middle_list = [_repeat_tokens(r[2]["middle"], cfg.n_samples).numpy().astype(np.int64) for r in records]
        real_bottom_list = [_repeat_tokens(r[2]["bottom"], cfg.n_samples).numpy().astype(np.int64) for r in records]
        timing_list = [_timing_from_payload(r[2], cfg.n_samples) for r in records]

        print(f'Full-length conditioning on {payload["source_stem"]} with {len(records)} bottom windows')
        generated_list = TransformerTokenSampler.generate_level_windows(
            prior=prior,
            seq_len=int(prior_cfg["model"]["inferred_seq_lens"]["bottom"]),
            num_samples=int(cfg.n_samples),
            start_frames=start_frames,
            device=device,
            temperature=float(cfg.temperature),
            top_k=cfg.top_k if cfg.top_k and cfg.top_k > 0 else None,
            upper_tokens_list=middle_list,
            second_upper_tokens_list=top_list if getattr(prior, "second_conditioner", None) is not None else None,
            timing_list=timing_list,
            level_name="bottom",
            progress_interval=int(cfg.progress_interval),
            level_time_frames=level_tf,
            level_grid=grid,
            use_overlap_prefixes=True,
        )
        validate_window_prefixes(generated_list, start_frames, level_tf, grid, "bottom")

        full_generated = assemble_token_timeline(generated_list, start_frames, level_tf, grid, total_frames).astype(np.int64)
        full_real_bottom = assemble_token_timeline(real_bottom_list, start_frames, level_tf, grid, total_frames).astype(np.int64)
        np.save(os.path.join(out, "generated_bottom_full.npy"), full_generated)
        np.save(os.path.join(out, "real_bottom_full.npy"), full_real_bottom)
        np.save(os.path.join(out, "start_frames.npy"), np.asarray(start_frames, dtype=np.int64))
        _save_window_npz(os.path.join(out, "generated_bottom_windows.npz"), [x.astype(np.int64) for x in generated_list])
        _save_window_npz(os.path.join(out, "real_bottom_windows.npz"), real_bottom_list)

        if vqvae_path:
            full_tensor = torch.from_numpy(full_generated)
            full_grid = dynamic_grid_for_tokens(full_tensor, grid)
            TransformerTokenDecoder.decode(
                "bottom",
                TransformerDecodeTarget(
                    tokens=full_tensor,
                    grid=full_grid,
                    chunk_time_cols=int(grid[0]) if isinstance(grid, list) and len(grid) == 2 else None,
                    trim_frames=total_frames,
                ),
                vqvae_path,
                weights,
                self.audio_config,
                out,
                device,
                decode_context_cols=max(1, int(grid[0]) // 2) if isinstance(grid, list) and len(grid) == 2 else 0,
            )
        print(f"Saved full-length bottom-only outputs to {out}")
        return EvaluationResult(output_dir=out)


class MiddleBottomConditionedPriorEvaluator:
    """!
    @brief Samples middle from real top codes, then bottom from generated middle codes.
    """

    def __init__(self, config: MiddleBottomConditionedPriorConfig, audio_config: Optional[AudioInversionConfig] = None):
        """!
        @brief Initialize evaluator.
        @param config Evaluation settings.
        @param audio_config Audio inversion settings.
        """
        self.config = config
        self.audio_config = audio_config or AudioInversionConfig(method="gradient", use_fixed_db_scale=True)

    def run(self) -> EvaluationResult:
        """!
        @brief Run middle-bottom conditioned prior evaluation.
        @return Evaluation artifact paths.
        """
        cfg = self.config
        set_global_seed(cfg.seed, deterministic=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        middle_prior, middle_cfg, _ = TransformerPriorLoader.load_prior("middle", cfg.middle_prior, device, cfg.weights_file)
        bottom_prior, bottom_cfg, _ = TransformerPriorLoader.load_prior("bottom", cfg.bottom_prior, device, cfg.weights_file)
        qfile = _pick_quantized_file(cfg.data_root, cfg.file)
        payload = torch.load(qfile, map_location="cpu", weights_only=False)
        if payload.get("format") != "windowed_v1":
            raise ValueError("Use *_window_quantized.pt files")

        out = os.path.join(os.path.abspath(os.path.expanduser(cfg.output_root)), datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(out, exist_ok=True)
        bottom_vqvae = cfg.bottom_vqvae or bottom_cfg.get("vqvae", {}).get("bottom_model_dir")
        weights = bottom_cfg.get("vqvae", {}).get("weights_file", "best_model.pth")
        middle_grid = middle_cfg["model"].get("inferred_grids", {}).get("middle")
        bottom_grid = bottom_cfg["model"].get("inferred_grids", {}).get("bottom")

        if not cfg.full_length:
            return self._run_single_window(middle_prior, middle_cfg, bottom_prior, bottom_cfg, payload, qfile, bottom_vqvae, weights, middle_grid, bottom_grid, out, device)
        return self._run_full_length(middle_prior, middle_cfg, bottom_prior, bottom_cfg, payload, bottom_vqvae, weights, middle_grid, bottom_grid, out, device)

    def _run_single_window(self, middle_prior, middle_cfg, bottom_prior, bottom_cfg, payload, qfile, bottom_vqvae, weights, middle_grid, bottom_grid, out, device) -> EvaluationResult:
        cfg = self.config
        top = _repeat_tokens(payload["top"], cfg.n_samples).to(device)
        real_middle = _repeat_tokens(payload["middle"], cfg.n_samples)
        real_bottom = _repeat_tokens(payload["bottom"], cfg.n_samples)
        timing = _timing_from_payload(payload, cfg.n_samples).to(device=device, dtype=torch.float32)
        print(f"Conditioning on {qfile}")
        with torch.no_grad():
            generated_middle = middle_prior.generate(
                batch_size=int(cfg.n_samples),
                start_tokens=None,
                upper_indices=top,
                seq_len=int(middle_cfg["model"]["inferred_seq_lens"]["middle"]),
                temperature=float(cfg.temperature),
                top_k=cfg.top_k if cfg.top_k and cfg.top_k > 0 else None,
                device=device,
                timing=timing,
                progress_label="middle",
                progress_interval=int(cfg.progress_interval),
            )
            generated_bottom = bottom_prior.generate(
                batch_size=int(cfg.n_samples),
                start_tokens=None,
                upper_indices=generated_middle,
                second_upper_indices=top if getattr(bottom_prior, "second_conditioner", None) is not None else None,
                seq_len=int(bottom_cfg["model"]["inferred_seq_lens"]["bottom"]),
                temperature=float(cfg.temperature),
                top_k=cfg.top_k if cfg.top_k and cfg.top_k > 0 else None,
                device=device,
                timing=timing,
                progress_label="bottom",
                progress_interval=int(cfg.progress_interval),
            ).cpu()
        generated_middle = generated_middle.cpu()
        for name, arr in [
            ("real_top", top.cpu().numpy().astype(np.int64)),
            ("generated_middle", generated_middle.numpy().astype(np.int64)),
            ("real_middle", real_middle.numpy().astype(np.int64)),
            ("generated_bottom", generated_bottom.numpy().astype(np.int64)),
            ("real_bottom", real_bottom.numpy().astype(np.int64)),
        ]:
            np.save(os.path.join(out, f"{name}.npy"), arr)
        if bottom_vqvae:
            TransformerTokenDecoder.decode(
                "bottom",
                TransformerDecodeTarget(tokens=generated_bottom, grid=bottom_grid),
                bottom_vqvae,
                weights,
                self.audio_config,
                out,
                device,
            )
        print(f"Saved middle->bottom ablation outputs to {out}")
        return EvaluationResult(output_dir=out)

    def _run_full_length(self, middle_prior, middle_cfg, bottom_prior, bottom_cfg, payload, bottom_vqvae, weights, middle_grid, bottom_grid, out, device) -> EvaluationResult:
        cfg = self.config
        records = _bottom_eligible_records(cfg.data_root, payload["source_stem"])
        ds_cfg = bottom_cfg.get("dataset", {}).get("level_target_time_frames", {})
        middle_tf = int(ds_cfg.get("middle", 512))
        bottom_tf = int(ds_cfg.get("bottom", 128))
        total_frames = int(records[0][2].get("total_frames", bottom_tf))
        start_frames = [r[0] for r in records]
        top_list = [_repeat_tokens(r[2]["top"], cfg.n_samples).numpy().astype(np.int64) for r in records]
        real_middle_list = [_repeat_tokens(r[2]["middle"], cfg.n_samples).numpy().astype(np.int64) for r in records]
        real_bottom_list = [_repeat_tokens(r[2]["bottom"], cfg.n_samples).numpy().astype(np.int64) for r in records]
        timing_list = [_timing_from_payload(r[2], cfg.n_samples) for r in records]

        print(f"Full-length conditioning on {payload['source_stem']} with {len(records)} bottom windows")
        generated_middle_list = TransformerTokenSampler.generate_level_windows(
            prior=middle_prior,
            seq_len=int(middle_cfg["model"]["inferred_seq_lens"]["middle"]),
            num_samples=int(cfg.n_samples),
            start_frames=start_frames,
            device=device,
            temperature=float(cfg.temperature),
            top_k=cfg.top_k if cfg.top_k and cfg.top_k > 0 else None,
            upper_tokens_list=top_list,
            timing_list=timing_list,
            level_name="middle",
            progress_interval=int(cfg.progress_interval),
            level_time_frames=middle_tf,
            level_grid=middle_grid,
            use_overlap_prefixes=True,
        )
        validate_window_prefixes(generated_middle_list, start_frames, middle_tf, middle_grid, "middle")

        generated_bottom_list = TransformerTokenSampler.generate_level_windows(
            prior=bottom_prior,
            seq_len=int(bottom_cfg["model"]["inferred_seq_lens"]["bottom"]),
            num_samples=int(cfg.n_samples),
            start_frames=start_frames,
            device=device,
            temperature=float(cfg.temperature),
            top_k=cfg.top_k if cfg.top_k and cfg.top_k > 0 else None,
            upper_tokens_list=[x.astype(np.int64) for x in generated_middle_list],
            second_upper_tokens_list=top_list if getattr(bottom_prior, "second_conditioner", None) is not None else None,
            timing_list=timing_list,
            level_name="bottom",
            progress_interval=int(cfg.progress_interval),
            level_time_frames=bottom_tf,
            level_grid=bottom_grid,
            use_overlap_prefixes=True,
        )
        validate_window_prefixes(generated_bottom_list, start_frames, bottom_tf, bottom_grid, "bottom")

        full_generated_middle = assemble_token_timeline(generated_middle_list, start_frames, middle_tf, middle_grid, total_frames).astype(np.int64)
        full_real_middle = assemble_token_timeline(real_middle_list, start_frames, middle_tf, middle_grid, total_frames).astype(np.int64)
        full_generated_bottom = assemble_token_timeline(generated_bottom_list, start_frames, bottom_tf, bottom_grid, total_frames).astype(np.int64)
        full_real_bottom = assemble_token_timeline(real_bottom_list, start_frames, bottom_tf, bottom_grid, total_frames).astype(np.int64)
        np.save(os.path.join(out, "generated_middle_full.npy"), full_generated_middle)
        np.save(os.path.join(out, "real_middle_full.npy"), full_real_middle)
        np.save(os.path.join(out, "generated_bottom_full.npy"), full_generated_bottom)
        np.save(os.path.join(out, "real_bottom_full.npy"), full_real_bottom)
        np.save(os.path.join(out, "start_frames.npy"), np.asarray(start_frames, dtype=np.int64))
        _save_window_npz(os.path.join(out, "generated_middle_windows.npz"), [x.astype(np.int64) for x in generated_middle_list])
        _save_window_npz(os.path.join(out, "real_middle_windows.npz"), real_middle_list)
        _save_window_npz(os.path.join(out, "generated_bottom_windows.npz"), [x.astype(np.int64) for x in generated_bottom_list])
        _save_window_npz(os.path.join(out, "real_bottom_windows.npz"), real_bottom_list)

        if bottom_vqvae:
            full_tensor = torch.from_numpy(full_generated_bottom)
            full_grid = dynamic_grid_for_tokens(full_tensor, bottom_grid)
            TransformerTokenDecoder.decode(
                "bottom",
                TransformerDecodeTarget(
                    tokens=full_tensor,
                    grid=full_grid,
                    chunk_time_cols=int(bottom_grid[0]) if isinstance(bottom_grid, list) and len(bottom_grid) == 2 else None,
                    trim_frames=total_frames,
                ),
                bottom_vqvae,
                weights,
                self.audio_config,
                out,
                device,
                decode_context_cols=max(1, int(bottom_grid[0]) // 2) if isinstance(bottom_grid, list) and len(bottom_grid) == 2 else 0,
            )
        print(f"Saved full-length middle->bottom ablation outputs to {out}")
        return EvaluationResult(output_dir=out)


def _bottom_eligible_records(data_root: str, source_stem: str):
    records = []
    for path in _source_files(data_root, source_stem):
        item = torch.load(path, map_location="cpu", weights_only=False)
        if item.get("format") != "windowed_v1":
            continue
        if "bottom" not in item.get("eligible_levels", []):
            continue
        records.append((int(item.get("start_frame", 0)), path, item))
    if not records:
        raise ValueError(f"No bottom-eligible windowed records found for {source_stem}")
    records.sort(key=lambda x: x[0])
    return records
