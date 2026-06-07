from __future__ import annotations

from typing import Optional

from evaluation.base_evaluators import PriorEvaluator
from evaluation.core import EvaluationResult
from generation.audio_inversion import AudioInversionConfig

from evaluation.transformer_prior import TransformerPriorSamplingConfig, run_transformer_prior_sampling


class TransformerPriorSamplingEvaluator(PriorEvaluator):
    """!
    @brief Orchestrates Transformer prior sampling, decoding, and artifact saving.
    """

    run_name = "transformer_prior"

    def __init__(self, config: TransformerPriorSamplingConfig, audio_config: Optional[AudioInversionConfig] = None):
        """!
        @brief Initialize evaluator.
        @param config Sampling configuration.
        @param audio_config Audio inversion configuration.
        """
        super().__init__(config, audio_config or AudioInversionConfig(method="gradient", use_fixed_db_scale=True))

    def _run_prior_artifacts(self, device) -> EvaluationResult:
        """!
        @brief Delegate the large transformer sampling flow to the existing implementation.
        """
        output_dir = run_transformer_prior_sampling(
            top_prior_path=self.config.top_prior_path,
            middle_prior_path=self.config.middle_prior_path,
            bottom_prior_path=self.config.bottom_prior_path,
            bottom_vqvae_path=self.config.bottom_vqvae_path,
            audio_method=self.audio_config.method,
            num_samples=int(self.config.n_samples),
            temperature=float(self.config.temperature),
            top_k=self.config.top_k,
            weights_file=self.config.weights_file,
            full_length=bool(self.config.full_length),
            full_length_overlap_fraction=float(self.config.full_length_overlap_fraction),
            seed=self.config.seed,
            timing_duration_seconds=float(self.config.timing_duration_seconds),
            min_max_values_path=self.audio_config.min_max_values_path,
            use_fixed_db_scale=self.audio_config.use_fixed_db_scale,
            fixed_min_db=self.audio_config.fixed_min_db,
            fixed_max_db=self.audio_config.fixed_max_db,
            audio_config=self.audio_config,
            output_root=self.config.save_root,
        )
        return EvaluationResult(output_dir=output_dir)
