from importlib import import_module


_EXPORT_MODULES = {
    "AudioExporter": ".audio",
    "BottomConditionedPriorConfig": ".conditioned_priors",
    "BottomConditionedPriorEvaluator": ".conditioned_priors",
    "EvaluationOutputConfig": ".core",
    "EvaluationResult": ".core",
    "EvaluationRun": ".core",
    "GenerationPayload": ".base_evaluators",
    "HierarchicalPixelCNNSamplingConfig": ".pixelcnn_evaluators",
    "HierarchicalPixelCNNSamplingEvaluator": ".pixelcnn_evaluators",
    "HierarchicalVQVAEReconstructionConfig": ".vae_evaluators",
    "HierarchicalVQVAEReconstructionEvaluator": ".vae_evaluators",
    "JukeboxVQVAEReconstructionConfig": ".vae_evaluators",
    "JukeboxVQVAEReconstructionEvaluator": ".vae_evaluators",
    "MaestroVQVAETestSetConfig": ".vqvae_testset_evaluators",
    "MaestroVQVAETestSetEvaluator": ".vqvae_testset_evaluators",
    "MiddleBottomConditionedPriorConfig": ".conditioned_priors",
    "MiddleBottomConditionedPriorEvaluator": ".conditioned_priors",
    "ModelLoader": ".model_loading",
    "PriorEvaluator": ".base_evaluators",
    "ReconstructionPayload": ".base_evaluators",
    "SinglePixelCNNSamplingConfig": ".pixelcnn_evaluators",
    "SinglePixelCNNSamplingEvaluator": ".pixelcnn_evaluators",
    "SingleVQVAEReconstructionConfig": ".vae_evaluators",
    "SingleVQVAEReconstructionEvaluator": ".vae_evaluators",
    "SpectrogramPlotConfig": ".visualization",
    "SpectrogramVisualizer": ".visualization",
    "TransformerDecodeTarget": ".transformer_prior",
    "TransformerPriorBundle": ".transformer_prior",
    "TransformerPriorLoader": ".transformer_prior",
    "TransformerPriorSamplingConfig": ".transformer_prior",
    "TransformerPriorSamplingEvaluator": ".transformer_evaluators",
    "TransformerTokenDecoder": ".transformer_prior",
    "TransformerTokenSampler": ".transformer_prior",
    "VQVAEEvaluator": ".base_evaluators",
    "VQVAEModelTestSpec": ".vqvae_testset_evaluators",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name):
    """!
    @brief Lazily import evaluation exports to avoid cross-package import cycles.
    @param name Exported object name.
    @return Requested exported object.
    """
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
