"""Compatibility wrappers for older PixelCNN evaluation imports.

The implementation moved to `evaluation.model_loading.ModelLoader`; this
module keeps existing scripts from breaking while the test CLIs migrate.
"""

from evaluation.model_loading import ModelLoader


resolve_model_paths = ModelLoader.resolve_model_paths
get_prior_names = ModelLoader.get_prior_names
parse_prior_arrays = ModelLoader.parse_prior_arrays
normalize_state_dict_keys = ModelLoader.normalize_state_dict_keys
infer_num_embeddings_from_state_dict = ModelLoader.infer_num_embeddings_from_state_dict
infer_two_level_conditioning_mode = ModelLoader.infer_two_level_conditioning_mode
load_hierarchical_pixelcnn_model = ModelLoader.load_hierarchical_pixelcnn_model
