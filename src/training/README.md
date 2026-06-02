# Training Infrastructure

The `src/training/` directory contains the shared training infrastructure used by the refactored training scripts. Its goal is to keep the training scripts thin and move repeated behavior, such as config handling, dataloaders, train/validation loops, checkpointing, resume, JSON loss history, plots, and sample callbacks, into reusable modules.

Each model still keeps its own YAML config file. The shared code expects the configs to use similar sections, mainly `task`, `dataset`, `model`, `training`, `resume`, and `callbacks`, while allowing each model to keep its architecture-specific fields under `model`.

## File Responsibilities

### `engine.py`

`engine.py` defines the generic training loop.

Main responsibilities:

- `TrainingEngine`: runs training end to end for a model, including train/validation tqdm bars, AMP, gradient accumulation, gradient clipping, scheduler stepping, checkpoint saving, best model saving, early stopping, loss history JSON, loss plots, resume, and optional sample callbacks.
- `TrainingAdapter`: the interface that each model family implements so the engine can train it without knowing model-specific details.
- `StepResult`: the object returned by each model's `train_step`/`val_step`, containing the scalar loss, metrics, and batch size.

The engine does not know whether it is training a VQ-VAE, hierarchical VQ-VAE, PixelCNN, or hierarchical PixelCNN. It only calls the adapter methods.

### `adapters.py`

`adapters.py` contains model-specific adapters used by the shared engine.

Main responsibilities:

- `SingleVQVAEAdapter`: builds and trains the single-level VQ-VAE.
- `TwoLevelVQVAEAdapter`: builds and trains the two-level hierarchical VQ-VAE.
- `SinglePixelCNNAdapter`: builds and trains the PixelCNN prior for a single VQ-VAE code stream.
- `TwoLevelPixelCNNAdapter`: builds and trains the hierarchical PixelCNN prior for top and bottom code streams.

Adapters are responsible for the parts that vary by model:

- Creating the model.
- Creating the optimizer, and optionally a scheduler.
- Preparing model-specific losses in `train_step`.
- Defining validation behavior when it differs from training.
- Adding extra checkpoint state, such as `data_variance`.
- Creating optional sample-generation callbacks.
- Naming latest, best, and periodic checkpoint files.

### `data_modules.py`

`data_modules.py` contains reusable dataset and dataloader setup.

Main responsibilities:

- `SpectrogramWindowDataModule`: loads precomputed image-like `.npy` windows for VQ-VAE training. In configs this is `dataset.input_mode: image`; the older `spectrogram` name is still accepted as an alias.
- `AudioWindowDataModule`: loads raw audio windows with `RawAudioWindowDataset`, then applies GPU audio-to-mel preprocessing and augmentation.
- `QuantizedPriorDataModule`: loads precomputed single-level or two-level VQ-VAE code datasets for PixelCNN training, including manifest filtering, train/test split merging, fixed validation parity, and even/odd train parity changes per epoch.
- `WindowParityEpochSampler`: changes the active train subset each epoch, for example alternating even and odd windows.
- `build_vqvae_data_module`: chooses audio or image VQ-VAE data setup from the config.

Data modules return a `DataBundle`, which is the common object passed into the engine and adapters.

### `common.py`

`common.py` stores shared utilities that do not belong to one model.

Main responsibilities:

- `DataBundle`: common container for dataloaders, datasets, preprocessing modules, variance, min/max values, file paths, embedding counts, and metadata.
- Config helpers: `get_training_cfg`, `get_callbacks_cfg`, and `get_resume_cfg`.
- Run artifact helpers: `create_run_dir`, `save_yaml`, `save_history`, and `plot_history`.
- Resume helpers: `normalize_history`, `load_history_file`, `best_metric_from_history`, and `historical_patience_counter`.
- Dataloader helpers: `dataloader_kwargs` and `move_to_device`.
- Audio preprocessing helpers: `estimate_preprocessed_variance`.
- Sample callback helpers: `collect_callback_samples` and `make_sample_generator`.

This file should stay model-agnostic. If a helper only makes sense for one model family, it usually belongs in an adapter or in a model-specific module.

### `runners.py`

`runners.py` connects configs, data modules, adapters, and the engine.

Main responsibilities:

- Load a YAML config.
- Normalize common config sections.
- Apply CLI overrides, such as `--input-mode` or `--resume-checkpoint`.
- Choose the correct data module.
- Choose the correct adapter.
- Start `TrainingEngine`.

The public runner functions are:

- `run_single_vqvae_training`
- `run_two_level_vqvae_training`
- `run_single_pixelcnn_training`
- `run_two_level_pixelcnn_training`

Training scripts should call these functions instead of creating models, datasets, or loops themselves.

### `jukebox_vqvae.py`

`jukebox_vqvae.py` contains the remaining Jukebox VQ-VAE training loop.

It is kept separate because Jukebox training still has more specialized behavior than the shared phase-1 engine, including level-specific model behavior and existing callback/checkpoint expectations. It reuses common helpers where possible, but it is not yet fully migrated to `TrainingEngine`.

This file replaces the old `train_vqvae_utils.py` module. The Jukebox script imports it directly as:

```python
from training.jukebox_vqvae import train_vqvae_jukebox
```

### `__init__.py`

`__init__.py` exposes the main shared engine, adapters, and data modules as package imports.

It intentionally does not export the Jukebox trainer, because importing `training` should stay lightweight and should not load Jukebox-specific dependencies unless they are actually needed.

## How Training Scripts Use This Directory

The refactored scripts under `src/train_scripts/` are now thin CLI wrappers:

- `train_vqvae_torch_script.py` calls `run_single_vqvae_training`.
- `train_vqvae_hierarchical.py` calls `run_two_level_vqvae_training`.
- `train_pixel_cnn.py` calls `run_single_pixelcnn_training`.
- `train_pixel_cnn_hierarchical.py` calls `run_two_level_pixelcnn_training`.
- `train_vqvae_jukebox.py` still calls `train_vqvae_jukebox` from `training/jukebox_vqvae.py`.

PixelCNN paths expect indices already precomputed by `src/processing/preprocess_vqvae_quantization.py`. Use `--variant single` for the single VQ-VAE PixelCNN and `--variant two_level` for the hierarchical VQ-VAE PixelCNN. The old PixelCNN path that quantized spectrograms inside the training loop was removed.

The usual flow is:

1. A training script parses CLI arguments.
2. The script calls a function in `runners.py`.
3. The runner loads and normalizes the YAML config.
4. The runner creates a data module and an adapter.
5. `TrainingEngine` runs the shared training lifecycle.
6. Artifacts are saved into the configured run directory.

## Where To Add New Behavior

- Add new model training behavior in `adapters.py`.
- Add new dataset loading behavior in `data_modules.py`.
- Add shared loop behavior in `engine.py`.
- Add reusable helpers in `common.py`.
- Add config/CLI wiring in `runners.py` and the corresponding `src/train_scripts/` wrapper.
- Keep specialized legacy Jukebox behavior in `jukebox_vqvae.py` until it is migrated to adapters.

## Config Boundary

Configs remain separate and model-specific:

- `config/config_vqvae.yaml`
- `config/config_vqvae_hierarchical.yaml`
- `config/config_pixelcnn.yaml`
- `config/config_pixelcnn_hierarchical.yaml`

They should use the same shared key names where possible, especially under `training`, `resume`, and `callbacks`. Architecture-specific differences should stay under `model`, and dataset-specific differences should stay under `dataset`.
