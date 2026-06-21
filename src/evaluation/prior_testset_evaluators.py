from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets.jukebox_precomputed_hierarchical_dataset import JukeboxQuantizedDataset
from datasets.quantized_dataset import PixelCNNQuantizedDataset, TwoLevelPixelCNNQuantizedDataset
from datasets.raw_audio_dataset import list_audio_files
from evaluation.base_evaluators import BaseEvaluator
from evaluation.core import EvaluationResult
from evaluation.model_loading import ModelLoader
from evaluation.transformer_prior import load_transformer_prior
from train_scripts.jukebox_utils import split_paths_by_maestro_metadata
from utils import list_npy_files, load_config


@dataclass
class PriorModelTestSpec:
    """!
    @brief Description of one autoregressive prior checkpoint to score.
    """

    name: str
    kind: str
    model_path: str
    weights_file: str = "best_model.pth"
    level: Optional[str] = None


@dataclass
class MaestroPriorTestSetConfig:
    """!
    @brief Settings for quantitative prior evaluation on held-out token data.
    """

    models: List[PriorModelTestSpec] = field(default_factory=list)
    split: str = "test"
    batch_size: int = 8
    max_samples: Optional[int] = None
    seed: int = 42
    window_parity: str = "all"
    pixelcnn_quantized_path: Optional[str] = None
    hierarchical_pixelcnn_quantized_path: Optional[str] = None
    transformer_quantized_path: Optional[str] = None
    processed_path: Optional[str] = None
    metadata_path: Optional[str] = None
    raw_path: Optional[str] = None
    save_root: str = "samples/maestro_prior_testset"


class TokenMetricAccumulator:
    """!
    @brief Accumulates token-level negative log-likelihood and accuracy.
    """

    def __init__(self, top_k: int = 5):
        """!
        @brief Initialize empty metric totals.
        @param top_k K used for top-k token accuracy.
        """
        self.top_k = int(top_k)
        self.loss_sum = 0.0
        self.token_count = 0
        self.correct_count = 0
        self.topk_correct_count = 0
        self.sample_count = 0

    def add(self, logits: torch.Tensor, target: torch.Tensor) -> Dict[str, np.ndarray]:
        """!
        @brief Add one logits/target batch to the accumulator.
        @param logits Class logits shaped `(B, K, ...)`.
        @param target Integer targets shaped `(B, ...)`.
        @return Per-sample metric arrays.
        """
        if logits.ndim < 3:
            raise ValueError(f"logits must have shape (B, K, ...), got {tuple(logits.shape)}")
        if target.shape[0] != logits.shape[0]:
            raise ValueError("logits and target batch sizes do not match.")

        losses = F.cross_entropy(logits, target.long(), reduction="none")
        batch_size = int(target.shape[0])
        flat_losses = losses.reshape(batch_size, -1)
        sample_loss_sum = flat_losses.sum(dim=1)
        sample_token_count = torch.full(
            (batch_size,),
            flat_losses.shape[1],
            dtype=torch.long,
            device=target.device,
        )

        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == target).reshape(batch_size, -1).sum(dim=1)
        k = min(self.top_k, int(logits.shape[1]))
        topk_correct = torch.zeros_like(correct)
        if k > 1:
            topk = torch.topk(logits, k=k, dim=1).indices
            topk_correct = (topk == target.unsqueeze(1)).any(dim=1).reshape(batch_size, -1).sum(dim=1)

        self.loss_sum += float(sample_loss_sum.sum().item())
        self.token_count += int(sample_token_count.sum().item())
        self.correct_count += int(correct.sum().item())
        self.topk_correct_count += int(topk_correct.sum().item())
        self.sample_count += batch_size

        loss_mean = sample_loss_sum / sample_token_count.to(dtype=sample_loss_sum.dtype)
        return {
            "nll": loss_mean.detach().cpu().numpy(),
            "bits_per_token": (loss_mean / math.log(2.0)).detach().cpu().numpy(),
            "perplexity": torch.exp(loss_mean).detach().cpu().numpy(),
            "accuracy": (correct.float() / sample_token_count.float()).detach().cpu().numpy(),
            "topk_accuracy": (topk_correct.float() / sample_token_count.float()).detach().cpu().numpy(),
            "token_count": sample_token_count.detach().cpu().numpy(),
        }

    def summary(self, prefix: str = "") -> Dict[str, float]:
        """!
        @brief Build aggregate metrics.
        @param prefix Optional key prefix.
        @return JSON-serializable summary dictionary.
        """
        if self.token_count <= 0:
            raise ValueError("No tokens were accumulated.")
        nll = self.loss_sum / self.token_count
        key = f"{prefix}_" if prefix else ""
        return {
            f"{key}num_samples": int(self.sample_count),
            f"{key}num_tokens": int(self.token_count),
            f"{key}nll": float(nll),
            f"{key}cross_entropy": float(nll),
            f"{key}bits_per_token": float(nll / math.log(2.0)),
            f"{key}perplexity": float(math.exp(nll)),
            f"{key}accuracy": float(self.correct_count / self.token_count),
            f"{key}top{self.top_k}_accuracy": float(self.topk_correct_count / self.token_count),
        }


class PriorTestSetRunner:
    """!
    @brief Base class for scoring one prior checkpoint.
    """

    def __init__(self, spec: PriorModelTestSpec, config: MaestroPriorTestSetConfig, device: torch.device):
        """!
        @brief Store shared runner state.
        @param spec Prior model specification.
        @param config Evaluation configuration.
        @param device Device used for model inference.
        """
        self.spec = spec
        self.config = config
        self.device = device
        self.run_config = load_config(BaseEvaluator._config_path_for(ModelLoader.model_reference(spec.model_path)))

    def evaluate(self) -> Dict:
        """!
        @brief Score a prior checkpoint.
        @return Dictionary with summary and per-sample rows.
        """
        raise NotImplementedError

    def _selected_indices(self, total: int) -> List[int]:
        """!
        @brief Return deterministic dataset indices after optional shuffling/capping.
        @param total Number of dataset examples.
        @return Selected dataset indices.
        """
        indices = np.arange(total)
        rng = np.random.default_rng(int(self.config.seed))
        rng.shuffle(indices)
        if self.config.max_samples is not None:
            indices = indices[:int(self.config.max_samples)]
        return [int(index) for index in indices]

    @staticmethod
    def _batches(indices: List[int], batch_size: int) -> List[List[int]]:
        """!
        @brief Split selected indices into batches.
        @param indices Dataset indices.
        @param batch_size Batch size.
        @return List of index batches.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}.")
        return [indices[start:start + batch_size] for start in range(0, len(indices), batch_size)]


class SinglePixelCNNTestSetRunner(PriorTestSetRunner):
    """!
    @brief Computes token NLL metrics for a single-level PixelCNN prior.
    """

    def evaluate(self) -> Dict:
        """!
        @brief Score a single-level PixelCNN on quantized test tokens.
        @return Summary and per-sample rows.
        """
        dataset_cfg = self.run_config.get("dataset", {})
        quantized_path = self.config.pixelcnn_quantized_path or dataset_cfg.get("quantized_path")
        if not quantized_path:
            raise ValueError("PixelCNN quantized path is required.")
        dataset = PixelCNNQuantizedDataset(
            quantized_path=quantized_path,
            split=self.config.split,
            manifest_file=dataset_cfg.get("manifest_file", "pixelcnn_quantized_manifest.jsonl"),
            preload=False,
        )
        selected = self._selected_dataset_indices(dataset)
        model = ModelLoader.load_single_pixelcnn(
            self.spec.model_path,
            self.device,
            dataset.num_embeddings,
            self.spec.weights_file,
        )
        model.eval()

        accumulator = TokenMetricAccumulator()
        rows = []
        with torch.no_grad():
            for batch_indices in tqdm(self._batches(selected, int(self.config.batch_size)), desc=self.spec.name):
                batch = torch.stack([dataset[index] for index in batch_indices], dim=0).to(self.device)
                logits = model(batch)
                logits = self._normalize_pixelcnn_logits(logits)
                sample_metrics = accumulator.add(logits, batch)
                rows.extend(self._rows(dataset, batch_indices, sample_metrics))

        summary = self._base_summary(accumulator.summary())
        summary["quantized_path"] = os.path.abspath(os.path.expanduser(quantized_path))
        return {"summary": summary, "rows": rows}

    def _selected_dataset_indices(self, dataset) -> List[int]:
        """!
        @brief Select dataset indices after applying optional window-parity filtering.
        @param dataset PixelCNN quantized dataset.
        @return Shuffled and optionally capped indices.
        """
        indices = self._filter_window_parity(dataset, list(range(len(dataset))))
        rng = np.random.default_rng(int(self.config.seed))
        rng.shuffle(indices)
        if self.config.max_samples is not None:
            indices = indices[:int(self.config.max_samples)]
        return [int(index) for index in indices]

    def _filter_window_parity(self, dataset, indices: List[int]) -> List[int]:
        """!
        @brief Apply optional even/odd overlap-window filtering.
        @param dataset PixelCNN quantized dataset.
        @param indices Preselected dataset indices.
        @return Filtered indices.
        """
        parity = str(self.config.window_parity or "all").strip().lower()
        if parity == "all" or not hasattr(dataset, "indices_for_window_parity"):
            return indices
        allowed = set(dataset.indices_for_window_parity(parity))
        return [index for index in indices if index in allowed]

    @staticmethod
    def _normalize_pixelcnn_logits(logits: torch.Tensor) -> torch.Tensor:
        """!
        @brief Convert PixelCNN logits to `(B, K, H, W)`.
        @param logits Raw model output.
        @return Normalized logits.
        """
        if logits.ndim == 5 and logits.shape[2] == 1:
            return logits.squeeze(2)
        if logits.ndim == 4:
            return logits
        raise ValueError(f"Expected PixelCNN logits with 4 or 5 dims, got {tuple(logits.shape)}")

    def _rows(self, dataset, batch_indices: List[int], sample_metrics: Dict[str, np.ndarray]) -> List[Dict]:
        """!
        @brief Build per-sample CSV rows for a PixelCNN batch.
        """
        rows = []
        for local_idx, dataset_idx in enumerate(batch_indices):
            rows.append(self._metric_row(dataset.files[dataset_idx], sample_metrics, local_idx))
        return rows

    def _metric_row(self, file_path: str, sample_metrics: Dict[str, np.ndarray], local_idx: int) -> Dict:
        """!
        @brief Build one per-sample metric row.
        """
        return {
            "model_name": self.spec.name,
            "kind": self.spec.kind,
            "level": self.spec.level or "",
            "file_path": file_path,
            "component": "tokens",
            "nll": float(sample_metrics["nll"][local_idx]),
            "cross_entropy": float(sample_metrics["nll"][local_idx]),
            "bits_per_token": float(sample_metrics["bits_per_token"][local_idx]),
            "perplexity": float(sample_metrics["perplexity"][local_idx]),
            "accuracy": float(sample_metrics["accuracy"][local_idx]),
            "top5_accuracy": float(sample_metrics["topk_accuracy"][local_idx]),
            "num_tokens": int(sample_metrics["token_count"][local_idx]),
        }

    def _base_summary(self, metrics: Dict) -> Dict:
        """!
        @brief Add model metadata to aggregate metrics.
        """
        return {
            "model_name": self.spec.name,
            "kind": self.spec.kind,
            "level": self.spec.level,
            "model_path": self.spec.model_path,
            "weights_file": self.spec.weights_file,
            "split": self.config.split,
            "window_parity": self.config.window_parity,
            **metrics,
        }


class HierarchicalPixelCNNTestSetRunner(SinglePixelCNNTestSetRunner):
    """!
    @brief Computes top, bottom, and combined metrics for a two-level PixelCNN prior.
    """

    def evaluate(self) -> Dict:
        """!
        @brief Score top and bottom hierarchical PixelCNN priors.
        @return Summary and per-sample rows.
        """
        dataset_cfg = self.run_config.get("dataset", {})
        quantized_path = self.config.hierarchical_pixelcnn_quantized_path or dataset_cfg.get("quantized_path")
        if not quantized_path:
            raise ValueError("Hierarchical PixelCNN quantized path is required.")
        dataset = TwoLevelPixelCNNQuantizedDataset(
            quantized_path=quantized_path,
            split=self.config.split,
            manifest_file=dataset_cfg.get("manifest_file", "pixelcnn_quantized_manifest.jsonl"),
            preload=False,
        )
        selected = self._selected_dataset_indices(dataset)
        model, _ = ModelLoader.load_hierarchical_pixelcnn_model(
            self.spec.model_path,
            self.device,
            self.spec.weights_file,
        )
        model.eval()

        top_accumulator = TokenMetricAccumulator()
        bottom_accumulator = TokenMetricAccumulator()
        rows = []
        with torch.no_grad():
            for batch_indices in tqdm(self._batches(selected, int(self.config.batch_size)), desc=self.spec.name):
                top = torch.stack([dataset[index][0] for index in batch_indices], dim=0).to(self.device)
                bottom = torch.stack([dataset[index][1] for index in batch_indices], dim=0).to(self.device)

                top_logits = self._normalize_pixelcnn_logits(model(top, level="top"))
                bottom_logits = self._normalize_pixelcnn_logits(model(bottom, cond=top, level="bottom"))
                top_metrics = top_accumulator.add(top_logits, top)
                bottom_metrics = bottom_accumulator.add(bottom_logits, bottom)
                rows.extend(self._hierarchical_rows(dataset, batch_indices, top_metrics, bottom_metrics))

        top_summary = top_accumulator.summary("top")
        bottom_summary = bottom_accumulator.summary("bottom")
        total_loss = top_accumulator.loss_sum + bottom_accumulator.loss_sum
        total_tokens = top_accumulator.token_count + bottom_accumulator.token_count
        total_nll = total_loss / total_tokens
        summary = self._base_summary(
            {
                **top_summary,
                **bottom_summary,
                "num_samples": int(top_accumulator.sample_count),
                "num_tokens": int(total_tokens),
                "nll": float(total_nll),
                "cross_entropy": float(total_nll),
                "bits_per_token": float(total_nll / math.log(2.0)),
                "perplexity": float(math.exp(total_nll)),
                "accuracy": float(
                    (top_accumulator.correct_count + bottom_accumulator.correct_count) / total_tokens
                ),
                "top5_accuracy": float(
                    (top_accumulator.topk_correct_count + bottom_accumulator.topk_correct_count) / total_tokens
                ),
            }
        )
        summary["quantized_path"] = os.path.abspath(os.path.expanduser(quantized_path))
        return {"summary": summary, "rows": rows}

    def _hierarchical_rows(
        self,
        dataset,
        batch_indices: List[int],
        top_metrics: Dict[str, np.ndarray],
        bottom_metrics: Dict[str, np.ndarray],
    ) -> List[Dict]:
        """!
        @brief Build top and bottom per-sample rows.
        """
        rows = []
        for local_idx, dataset_idx in enumerate(batch_indices):
            top_row = self._metric_row(dataset.files[dataset_idx], top_metrics, local_idx)
            bottom_row = self._metric_row(dataset.files[dataset_idx], bottom_metrics, local_idx)
            top_row["component"] = "top"
            bottom_row["component"] = "bottom"
            rows.append(top_row)
            rows.append(bottom_row)
        return rows


class TransformerPriorTestSetRunner(PriorTestSetRunner):
    """!
    @brief Computes next-token NLL metrics for one Jukebox Transformer prior level.
    """

    def __init__(self, spec: PriorModelTestSpec, config: MaestroPriorTestSetConfig, device: torch.device):
        """!
        @brief Load a Transformer prior and keep its saved config.
        """
        super().__init__(spec, config, device)
        if spec.level not in {"top", "middle", "bottom"}:
            raise ValueError("Transformer prior specs require level top, middle, or bottom.")
        self.model, self.run_config, self.loaded_model_path = load_transformer_prior(
            spec.level,
            spec.model_path,
            device,
            spec.weights_file,
        )

    def evaluate(self) -> Dict:
        """!
        @brief Score a Transformer prior on real quantized token sequences.
        @return Summary and per-sample rows.
        """
        dataset = self._dataset()
        selected = self._selected_indices(len(dataset))
        accumulator = TokenMetricAccumulator()
        rows = []
        self.model.eval()
        with torch.no_grad():
            for batch_indices in tqdm(self._batches(selected, int(self.config.batch_size)), desc=self.spec.name):
                batch = self._load_batch(dataset, batch_indices)
                logits, target = self._logits_and_target(batch)
                sample_metrics = accumulator.add(logits.transpose(1, 2), target)
                for local_idx, dataset_idx in enumerate(batch_indices):
                    rows.append(self._metric_row(dataset.files[dataset_idx], sample_metrics, local_idx))

        summary = {
            "model_name": self.spec.name,
            "kind": self.spec.kind,
            "level": self.spec.level,
            "model_path": self.spec.model_path,
            "loaded_model_path": self.loaded_model_path,
            "weights_file": self.spec.weights_file,
            "split": self.config.split,
            "window_parity": self.config.window_parity,
            "quantized_path": os.path.abspath(os.path.expanduser(self._quantized_path())),
            **accumulator.summary(),
        }
        return {"summary": summary, "rows": rows}

    def _dataset(self) -> JukeboxQuantizedDataset:
        """!
        @brief Build the held-out Jukebox token dataset for this Transformer level.
        @return Filtered JukeboxQuantizedDataset.
        """
        dataset_cfg = dict(self.run_config.get("dataset", {}))
        if self.config.metadata_path:
            dataset_cfg["metadata_path"] = self.config.metadata_path
        if self.config.raw_path:
            dataset_cfg["raw_path"] = self.config.raw_path
        if self.config.processed_path:
            dataset_cfg["processed_path"] = self.config.processed_path

        file_paths = self._split_file_paths(dataset_cfg)
        dataset = JukeboxQuantizedDataset(
            quantized_path=self._quantized_path(),
            file_paths=file_paths,
            target_time_frames=int(dataset_cfg.get("target_time_frames", 2048)),
            level_target_time_frames=dataset_cfg.get("level_target_time_frames") or {},
            selected_level=self.spec.level,
            sample_rate=int(dataset_cfg.get("sample_rate", 22050)),
            hop_length=int(dataset_cfg.get("hop_length", 256)),
            window_parity=self.config.window_parity,
            metadata_path=dataset_cfg.get("metadata_path"),
            key_infer_missing_mode_as=self.run_config.get("conditioning", {})
            .get("key", {})
            .get("infer_missing_mode_as", "major"),
            key_dropout_prob=0.0,
            timing_dropout_prob=0.0,
        )
        return dataset

    def _quantized_path(self) -> str:
        """!
        @brief Resolve the transformer quantized token directory.
        @return Quantized token path.
        """
        return self.config.transformer_quantized_path or self.run_config.get("dataset", {}).get(
            "quantized_data_path",
            "./data/processed/maestro_quantized/",
        )

    def _split_file_paths(self, dataset_cfg: dict) -> List[str]:
        """!
        @brief Resolve source files for the requested official MAESTRO split.
        @param dataset_cfg Dataset configuration.
        @return Source file paths for the requested split.
        """
        all_paths = list_npy_files(dataset_cfg.get("processed_path", ""))
        if not all_paths:
            audio_cfg = dataset_cfg.get("audio", {})
            raw_path = dataset_cfg.get("raw_path")
            all_paths = list_audio_files(raw_path, extensions=audio_cfg.get("extensions")) if raw_path else []
        if not all_paths:
            raise ValueError("Could not find processed spectrograms or raw audio paths for metadata split.")

        train_paths, validation_paths, test_paths = split_paths_by_maestro_metadata(all_paths, dataset_cfg)
        paths_by_split = {
            "train": train_paths,
            "validation": validation_paths,
            "test": test_paths,
        }
        return paths_by_split[self.config.split]

    def _load_batch(self, dataset: JukeboxQuantizedDataset, batch_indices: List[int]) -> Dict[str, Optional[torch.Tensor]]:
        """!
        @brief Load and stack one Transformer token batch.
        @param dataset Source token dataset.
        @param batch_indices Dataset indices in this batch.
        @return Batch dictionary on the evaluation device.
        """
        samples = [dataset[index] for index in batch_indices]
        targets = torch.stack([sample[0].reshape(-1) for sample in samples], dim=0).long().to(self.device)
        cond = self._stack_optional([sample[1] for sample in samples])
        second_cond = self._stack_optional([sample[2] for sample in samples])
        timing = torch.stack([sample[3] for sample in samples], dim=0).float().to(self.device)
        metadata = [sample[4] for sample in samples]
        key_ids = torch.stack([item["key_id"] for item in metadata], dim=0).long().to(self.device)
        timing_mask = torch.stack([item["timing_mask"] for item in metadata], dim=0).bool().to(self.device)
        return {
            "target": targets,
            "cond": cond,
            "second_cond": second_cond,
            "timing": timing,
            "key_ids": key_ids,
            "timing_mask": timing_mask,
        }

    def _stack_optional(self, tensors: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """!
        @brief Stack optional conditioning tensors.
        @param tensors Conditioning tensors from dataset samples.
        @return Flattened batch tensor or None.
        """
        if not tensors or tensors[0] is None or tensors[0].numel() == 0:
            return None
        return torch.stack([tensor.reshape(-1) for tensor in tensors], dim=0).long().to(self.device)

    def _logits_and_target(self, batch: Dict[str, Optional[torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        """!
        @brief Compute shifted next-token logits and targets for CE.
        @param batch Prepared Transformer batch.
        @return Tuple `(logits, target)` where logits are `(B, T, K)`.
        """
        indices = batch["target"]
        prepend_start_embedding = False
        if self.model.use_start_embedding:
            input_tokens = indices[:, :-1]
            target = indices
            prepend_start_embedding = True
        elif self.model.use_bos_token:
            bos = torch.full(
                (indices.shape[0], 1),
                self.model.bos_token_id,
                dtype=indices.dtype,
                device=indices.device,
            )
            input_tokens = torch.cat([bos, indices[:, :-1]], dim=1)
            target = indices
        else:
            input_tokens = indices[:, :-1]
            target = indices[:, 1:]

        logits = self.model(
            input_tokens,
            upper_indices=batch["cond"],
            second_upper_indices=batch["second_cond"],
            timing=batch["timing"],
            timing_mask=batch["timing_mask"],
            key_ids=batch["key_ids"],
            prepend_start_embedding=prepend_start_embedding,
        )
        if logits.shape[:2] != target.shape:
            raise ValueError(f"Transformer logits shape {tuple(logits.shape)} does not align with target {tuple(target.shape)}")
        return logits, target

    def _metric_row(self, file_path: str, sample_metrics: Dict[str, np.ndarray], local_idx: int) -> Dict:
        """!
        @brief Build one per-sample Transformer metric row.
        """
        return {
            "model_name": self.spec.name,
            "kind": self.spec.kind,
            "level": self.spec.level or "",
            "file_path": file_path,
            "component": "tokens",
            "nll": float(sample_metrics["nll"][local_idx]),
            "cross_entropy": float(sample_metrics["nll"][local_idx]),
            "bits_per_token": float(sample_metrics["bits_per_token"][local_idx]),
            "perplexity": float(sample_metrics["perplexity"][local_idx]),
            "accuracy": float(sample_metrics["accuracy"][local_idx]),
            "top5_accuracy": float(sample_metrics["topk_accuracy"][local_idx]),
            "num_tokens": int(sample_metrics["token_count"][local_idx]),
        }


class MaestroPriorTestSetEvaluator(BaseEvaluator):
    """!
    @brief Evaluate PixelCNN and Transformer priors with held-out token likelihood metrics.
    """

    run_name = "maestro_prior_testset"

    def run(self) -> EvaluationResult:
        """!
        @brief Run all configured prior test-set evaluations and save reports.
        @return Evaluation artifact paths.
        """
        split = str(self.config.split).strip().lower()
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"split must be one of train, validation, test; got {self.config.split}.")
        if not self.config.models:
            raise ValueError("At least one prior model spec is required.")

        run = self._create_run(
            int(self.config.max_samples or 0),
            run_name=self.run_name,
            seed=int(self.config.seed),
        )
        summaries = []
        rows = []
        device = self._device()
        for spec in self.config.models:
            result = self._runner_for(spec, device).evaluate()
            summaries.append(result["summary"])
            rows.extend(result["rows"])

        ranked = sorted(summaries, key=lambda item: item["nll"])
        self._save_csv(run.path("per_sample_prior_metrics.csv"), rows)
        run.save_json(
            "metrics.json",
            run.metadata_payload(
                {
                    "config": self._config_dict(),
                    "metric": "next_token_cross_entropy",
                    "split": split,
                    "summaries": summaries,
                    "ranked_by_nll": ranked,
                }
            ),
        )
        return run.result()

    def _runner_for(self, spec: PriorModelTestSpec, device: torch.device) -> PriorTestSetRunner:
        """!
        @brief Build the correct runner implementation for a prior spec.
        @param spec Prior model specification.
        @param device Evaluation device.
        @return Prior runner.
        """
        kind = str(spec.kind).strip().lower()
        if kind == "single_pixelcnn":
            return SinglePixelCNNTestSetRunner(spec, self.config, device)
        if kind == "hierarchical_pixelcnn":
            return HierarchicalPixelCNNTestSetRunner(spec, self.config, device)
        if kind == "transformer":
            return TransformerPriorTestSetRunner(spec, self.config, device)
        raise ValueError(f"Unsupported prior kind '{spec.kind}'.")

    @staticmethod
    def _save_csv(path: str, rows: List[Dict]) -> None:
        """!
        @brief Save per-sample prior metrics as CSV.
        @param path Destination CSV path.
        @param rows Per-sample metric rows.
        """
        fieldnames = [
            "model_name",
            "kind",
            "level",
            "file_path",
            "component",
            "nll",
            "cross_entropy",
            "bits_per_token",
            "perplexity",
            "accuracy",
            "top5_accuracy",
            "num_tokens",
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
