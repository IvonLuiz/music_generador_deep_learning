from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from callbacks import EarlyStopping
from utils import set_global_seed

from .common import (
    DataBundle,
    best_metric_from_history,
    create_run_dir,
    get_callbacks_cfg,
    get_resume_cfg,
    get_training_cfg,
    historical_patience_counter,
    load_history_file,
    normalize_history,
    plot_history,
    save_history,
    save_yaml,
)


@dataclass
class StepResult:
    """!
    @brief Output returned by a TrainingAdapter for one train or validation batch.

    @param loss Scalar tensor used for backpropagation and finite-value checks.
    @param metrics Named scalar metrics accumulated into loss_history.json.
    @param batch_size Number of samples represented by the metrics.
    """
    loss: torch.Tensor
    metrics: Dict[str, object]
    batch_size: int


class TrainingAdapter:
    """!
    @brief Per-model contract used by TrainingEngine.

    The engine owns the generic training lifecycle: epochs, AMP, gradient
    accumulation, checkpointing, resume, early stopping and history files. Each
    adapter only describes model-specific behavior such as model construction,
    optimizer selection and the loss computation for a batch.
    """

    latest_filename = 'latest_model.pth'
    best_filenames = ('best_model.pth',)
    checkpoint_prefix = 'model_epoch'
    monitor_key = 'val_total'

    def run_subdir(self, config: dict) -> Optional[str]:
        """!
        @brief Optional subdirectory inserted between training.save_dir and timestamp.
        """
        return None

    def build_model(self, config: dict, data: DataBundle, device: torch.device) -> torch.nn.Module:
        """!
        @brief Build the model for this training task.
        """
        raise NotImplementedError

    def build_optimizer(self, model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
        """!
        @brief Build the optimizer used by the shared engine.
        """
        training_cfg = get_training_cfg(config)
        return torch.optim.Adam(
            model.parameters(),
            lr=float(training_cfg['learning_rate']),
            weight_decay=float(training_cfg.get('weight_decay', 0.0)),
        )

    def build_scheduler(self, optimizer: torch.optim.Optimizer, config: dict, steps_per_epoch: int):
        """!
        @brief Optionally build a per-step scheduler.
        """
        return None

    def load_model_state(
        self,
        model: torch.nn.Module,
        checkpoint: dict,
        config: dict,
        device: torch.device,
    ) -> dict:
        """!
        @brief Load model weights from a checkpoint.

        @return Dict with optional reset_optimizer/reset_scheduler flags.
        """
        model.load_state_dict(checkpoint['model_state'])
        return {}

    def config_for_save(self, config: dict, data: DataBundle, model: torch.nn.Module) -> dict:
        """!
        @brief Return the config payload written to run_dir/config.yaml.
        """
        return config

    def autocast_dtype(self, device: torch.device):
        """!
        @brief Optional dtype override for AMP autocast.
        """
        return None

    def prepare_batch(self, batch, data: DataBundle, device: torch.device, training: bool):
        """!
        @brief Move a batch to device and apply the data module preprocessor.
        """
        from .common import move_to_device

        batch = move_to_device(batch, device)
        if data.batch_preprocessor is not None:
            batch = data.batch_preprocessor(batch, augment=training)
        return batch

    def train_step(self, model: torch.nn.Module, batch, data: DataBundle) -> StepResult:
        """!
        @brief Compute the training loss and metrics for one prepared batch.
        """
        raise NotImplementedError

    def val_step(self, model: torch.nn.Module, batch, data: DataBundle) -> StepResult:
        """!
        @brief Compute validation metrics for one prepared batch.
        """
        return self.train_step(model, batch, data)

    def checkpoint_extra_state(self, model: torch.nn.Module, data: DataBundle, config: dict) -> dict:
        """!
        @brief Return adapter-specific values to store in model checkpoints.
        """
        return {}

    def create_sample_callback(
        self,
        model: torch.nn.Module,
        data: DataBundle,
        run_dir: str,
        device: torch.device,
        config: dict,
    ):
        """!
        @brief Optionally create a callback that writes visual/audio samples.
        """
        return None


def _metric_to_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    return float(value)


def _make_scaler(use_amp: bool):
    if not use_amp:
        return None
    try:
        return torch.amp.GradScaler('cuda')
    except TypeError:
        from torch.amp.grad_scaler import GradScaler

        return GradScaler(enabled=True)


def _autocast(device: torch.device, enabled: bool, dtype=None):
    kwargs = {'device_type': device.type, 'enabled': enabled}
    if dtype is not None:
        kwargs['dtype'] = dtype
    return torch.amp.autocast(**kwargs)


class TrainingEngine:
    """!
    @brief Shared training loop for VQ-VAE and PixelCNN style experiments.

    The engine intentionally does not know the model family being trained. It
    receives a TrainingAdapter plus a data module and handles the repeated
    training logic shared across implementations. This includes directory
    creation, config copy, train/validation progress bars, AMP, gradient
    accumulation, clipping, checkpointing, resume, loss JSON, plots and
    early stopping.
    """

    def __init__(
        self,
        config: dict,
        adapter: TrainingAdapter,
        data_module,
        config_path: Optional[str] = None,
    ):
        self.config = config
        self.adapter = adapter
        self.data_module = data_module
        self.config_path = config_path

    def run(self) -> str:
        """!
        @brief Execute training and return the artifact directory path.
        """
        training_cfg = get_training_cfg(self.config)
        callbacks_cfg = get_callbacks_cfg(self.config)
        resume_cfg = get_resume_cfg(self.config)

        seed = training_cfg.get('seed', 42)
        set_global_seed(int(seed))

        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Training on {device}')
        if torch.cuda.is_available():
            print('GPU:', torch.cuda.get_device_name(0))
            print('Capability:', torch.cuda.get_device_capability(0))
            print('CUDA memory allocated (MB):', round(torch.cuda.memory_allocated(0) / 1024**2, 2))

        data = self.data_module.setup(device)
        if data.batch_preprocessor is not None:
            data.batch_preprocessor.to(device)

        run_dir = create_run_dir(
            training_cfg['save_dir'],
            self.adapter.run_subdir(self.config),
        )
        self.config.setdefault('training', {})['seed'] = seed
        if self.config_path:
            self.config.setdefault('training', {})['config_path'] = self.config_path

        model = self.adapter.build_model(self.config, data, device).to(device)
        self.config = self.adapter.config_for_save(self.config, data, model)
        save_yaml(self.config, os.path.join(run_dir, 'config.yaml'))
        optimizer = self.adapter.build_optimizer(model, self.config)

        epochs = int(training_cfg['epochs'])
        batch_size = int(training_cfg['batch_size'])
        grad_accum_steps = int(training_cfg.get('gradient_accumulation_steps', 1))
        if grad_accum_steps < 1:
            raise ValueError(f'gradient_accumulation_steps must be >= 1, got {grad_accum_steps}')
        effective_batch = batch_size * grad_accum_steps
        print(
            f'Training parameters: batch_size={batch_size}, '
            f'grad_accum_steps={grad_accum_steps}, effective_batch={effective_batch}, '
            f'learning_rate={float(training_cfg["learning_rate"]):.6g}, epochs={epochs}, seed={seed}'
        )
        print(f'Saving artifacts to: {run_dir}')

        steps_per_epoch = int(data.metadata.get(
            'optimizer_steps_per_epoch',
            max(1, int(np.ceil(len(data.train_loader) / grad_accum_steps))),
        ))
        scheduler = self.adapter.build_scheduler(optimizer, self.config, steps_per_epoch)

        use_amp = bool(training_cfg.get('amp', True)) and device.type == 'cuda'
        autocast_dtype = self.adapter.autocast_dtype(device)
        scaler = _make_scaler(use_amp)
        max_grad_norm = training_cfg.get('max_grad_norm', 1.0)
        max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)

        # logic to resume training with some parameters options
        resume_checkpoint = resume_cfg.get('checkpoint_path')
        resume_enabled = bool(resume_cfg.get('enabled', False)) or bool(resume_checkpoint)
        reset_optimizer = bool(resume_cfg.get('reset_optimizer', False))
        reset_scheduler = bool(resume_cfg.get('reset_scheduler', reset_optimizer))
        history = {}
        start_epoch = 1
        best_metric = float('inf') # default to inf

        if resume_enabled:
            # update parameters to match the checkpoint if resuming
            history, start_epoch, best_metric = self._resume_training(
                model, resume_checkpoint, device, optimizer, scheduler, reset_optimizer, reset_scheduler)

            # skip training if config epochs is less than the checkpoint epoch
            if start_epoch > epochs:
                print(f'Checkpoint already completed epoch {start_epoch - 1}; configured epochs={epochs}.')
                save_history(history, run_dir)
                plot_history(history, run_dir)
                return run_dir

        history = normalize_history(history)

        # early stopping
        early_stopping = None
        if data.val_loader is not None:
            counter = historical_patience_counter(history, self.adapter.monitor_key, best_metric)
            early_stopping = EarlyStopping(
                patience=int(callbacks_cfg.get('early_stopping_patience')),
                verbose=True,
                best_score=-best_metric if best_metric is not None else None,
                counter=counter,
            )

        # logic to conditionally create a sample callback that generates audio/visuals during training
        sample_callback = None
        if bool(callbacks_cfg.get('save_samples', False)):
            sample_callback = self.adapter.create_sample_callback(model, data, run_dir, device, self.config)

        latest_path = os.path.join(run_dir, self.adapter.latest_filename)
        save_every = int(callbacks_cfg.get('save_every', 0) or 0)

        for epoch in range(start_epoch, epochs + 1):
            train_sampler = getattr(data.train_loader, 'sampler', None)
            if hasattr(train_sampler, 'set_epoch'):
                train_sampler.set_epoch(epoch - 1)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_metrics = self._run_train_epoch(
                epoch=epoch,
                epochs=epochs,
                model=model,
                data=data,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                use_amp=use_amp,
                autocast_dtype=autocast_dtype,
                grad_accum_steps=grad_accum_steps,
                max_grad_norm=max_grad_norm,
            )

            val_metrics = {}
            if data.val_loader is not None:
                model.eval()
                val_metrics = self._run_val_epoch(
                    epoch=epoch,
                    epochs=epochs,
                    model=model,
                    data=data,
                    use_amp=use_amp,
                    autocast_dtype=autocast_dtype,
                )

            self._append_history(history, train_metrics, val_metrics)
            save_history(history, run_dir)
            plot_history(history, run_dir)

            # The public monitor key is normalized to "total" here because
            # _append_history stores validation metrics with a val_ prefix.
            monitored = val_metrics.get('total', train_metrics.get('total'))
            metric_value = float(monitored) if monitored is not None else float('inf')
            checkpoint_payload = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
                'config': self.config,
                'epoch': epoch,
                'train_loss': float(train_metrics.get('total', metric_value)),
                'val_loss': None if not val_metrics else float(val_metrics.get('total', metric_value)),
                'metric_value': metric_value,
                'history': history,
            }
            checkpoint_payload.update(self.adapter.checkpoint_extra_state(model, data, self.config))
            torch.save(checkpoint_payload, latest_path)

            if save_every > 0 and (epoch % save_every == 0 or epoch == epochs):
                periodic_name = f'{self.adapter.checkpoint_prefix}_{epoch}.pth'
                torch.save(checkpoint_payload, os.path.join(run_dir, periodic_name))

            if metric_value < best_metric:
                best_metric = metric_value
                for filename in self.adapter.best_filenames:
                    torch.save(checkpoint_payload, os.path.join(run_dir, filename))
                print(f'New best metric: {best_metric:.4f}. Saved best model.')

            self._print_epoch_summary(epoch, epochs, train_metrics, val_metrics)

            if sample_callback is not None:
                sample_callback.step(epoch)

            if early_stopping is not None:
                early_stopping(metric_value)
                if early_stopping.early_stop:
                    print('Early stopping triggered.')
                    break

            if device.type == 'cuda':
                torch.cuda.empty_cache()

        return run_dir

    def _run_train_epoch(
        self,
        epoch: int,
        epochs: int,
        model,
        data: DataBundle,
        optimizer,
        scheduler,
        scaler,
        use_amp: bool,
        autocast_dtype,
        grad_accum_steps: int,
        max_grad_norm: Optional[float],
    ) -> Dict[str, float]:
        """!
        @brief Run one training epoch and return sample-weighted metric means.
        
        Logic handles gradient accumulation and AMP scaling. Metrics returned by the adapter are
        expected to be per-batch averages, and are weighted by batch size to compute epoch means.
        """
        totals = {}
        total_samples = 0
        skipped = 0
        desc_suffix = ''
        sampler = getattr(data.train_loader, 'sampler', None)
        if getattr(sampler, 'current_parity', None):
            desc_suffix = f':{sampler.current_parity}'

        progress = tqdm(data.train_loader, desc=f'Epoch {epoch:03d}/{epochs} [Train{desc_suffix}]')
        num_batches = len(data.train_loader)
        remainder = num_batches % grad_accum_steps
        first_remainder_step = num_batches - remainder + 1 if remainder else None

        for step, raw_batch in enumerate(progress, start=1):
            batch = self.adapter.prepare_batch(raw_batch, data, next(model.parameters()).device, training=True)
            # Scale the last partial accumulation window by its real size; this
            # keeps the final optimizer step from being underweighted.
            current_accum_steps = (
                remainder
                if first_remainder_step is not None and step >= first_remainder_step
                else grad_accum_steps
            )
            with _autocast(next(model.parameters()).device, use_amp, autocast_dtype):
                result = self.adapter.train_step(model, batch, data)    # this step is implemented by the child classes
                loss = result.loss / current_accum_steps

            if not torch.isfinite(result.loss):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # optimizer step if we've accumulated enough or if it's the last batch
            should_step = step % grad_accum_steps == 0 or step == num_batches
            if should_step:
                if scaler is not None:
                    if max_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            batch_size = max(1, int(result.batch_size))
            total_samples += batch_size
            for key, value in result.metrics.items():
                totals[key] = totals.get(key, 0.0) + _metric_to_float(value) * batch_size
            if total_samples:
                progress.set_postfix(loss=totals.get('total', 0.0) / total_samples)

        if skipped:
            print(f'Epoch {epoch}: skipped {skipped} non-finite training batches.')
        if total_samples == 0:
            raise RuntimeError(f'Epoch {epoch}: no valid training batches.')
        return {key: value / total_samples for key, value in totals.items()}

    def _run_val_epoch(
        self,
        epoch: int,
        epochs: int,
        model,
        data: DataBundle,
        use_amp: bool,
        autocast_dtype=None,
    ) -> Dict[str, float]:
        """!
        @brief Run one validation epoch and return sample-weighted metric means.
        """
        totals = {}
        total_samples = 0
        skipped = 0
        device = next(model.parameters()).device
        with torch.no_grad():
            progress = tqdm(data.val_loader, desc=f'Epoch {epoch:03d}/{epochs} [Val]')
            for raw_batch in progress:
                batch = self.adapter.prepare_batch(raw_batch, data, device, training=False)
                with _autocast(device, use_amp, autocast_dtype):
                    result = self.adapter.val_step(model, batch, data)
                if not torch.isfinite(result.loss):
                    skipped += 1
                    continue
                batch_size = max(1, int(result.batch_size))
                total_samples += batch_size
                for key, value in result.metrics.items():
                    totals[key] = totals.get(key, 0.0) + _metric_to_float(value) * batch_size
                if total_samples:
                    progress.set_postfix(val_loss=totals.get('total', 0.0) / total_samples)
        if skipped:
            print(f'Epoch {epoch}: skipped {skipped} non-finite validation batches.')
        if total_samples == 0:
            return {'total': float('nan')}
        return {key: value / total_samples for key, value in totals.items()}

    @staticmethod
    def _append_history(history: dict, train_metrics: dict, val_metrics: dict) -> None:
        """!
        @brief Append epoch metrics using both detailed and legacy-compatible keys.
        """
        for key, value in train_metrics.items():
            history.setdefault(key, []).append(float(value))
        if 'total' in train_metrics:
            history.setdefault('train_loss', []).append(float(train_metrics['total']))
        for key, value in val_metrics.items():
            history.setdefault(f'val_{key}', []).append(float(value))
        if 'total' in val_metrics:
            history.setdefault('val_loss', []).append(float(val_metrics['total']))

    @staticmethod
    def _print_epoch_summary(epoch: int, epochs: int, train_metrics: dict, val_metrics: dict) -> None:
        """!
        @brief Print a compact epoch summary after checkpoints are updated.
        """
        train_parts = ', '.join(f'{key} {value:.6f}' for key, value in train_metrics.items())
        val_parts = ', '.join(f'val_{key} {value:.6f}' for key, value in val_metrics.items())
        suffix = f'; {val_parts}' if val_parts else ''
        print(f'Epoch {epoch:03d}/{epochs} - {train_parts}{suffix}')

    def _resume_training(
        self,
        model: torch.nn.Module,
        resume_checkpoint: Optional[str] = None,
        device: torch.device = torch.device('cpu'),
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[object] = None,
        reset_optimizer: bool = False,
        reset_scheduler: bool = False,
    ) -> tuple[dict, int, float]:
        """!
        @brief Load model and training state from a checkpoint for resume functionality.
        """
        if not resume_checkpoint:
            raise ValueError('resume.enabled is true but resume.checkpoint_path is empty.')
        if not os.path.isfile(resume_checkpoint):
            raise FileNotFoundError(f'Resume checkpoint not found: {resume_checkpoint}')

        # checkpoint should have model_state, optimizer_state, scheduler_state,
        # epoch, train_loss, val_loss, metric_value, and history keys
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)

        # reset optimizer/scheduler if specified by config
        load_result = self.adapter.load_model_state(model, checkpoint, self.config, device) or {}
        if load_result.get('reset_optimizer'):
            reset_optimizer = True
        if load_result.get('reset_scheduler'):
            reset_scheduler = True
        if not reset_optimizer and checkpoint.get('optimizer_state') is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        if scheduler is not None and not reset_scheduler and checkpoint.get('scheduler_state') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state'])

        # history
        history = normalize_history(checkpoint.get('history', {}))
        file_history = load_history_file(os.path.dirname(resume_checkpoint))
        if file_history:
            history = file_history
        if 'epoch' in checkpoint:
            start_epoch = int(checkpoint['epoch']) + 1
        else:
            start_epoch = len(history.get(self.adapter.monitor_key, [])) + 1

        # best metric to continue track
        previous_best_metric = best_metric_from_history(
            history,
            self.adapter.monitor_key,
        )
        if previous_best_metric is None and checkpoint.get('metric_value') is not None:
            previous_best_metric = float(checkpoint['metric_value'])
        if previous_best_metric is None:
            raise ValueError('Failed to determine best metric from checkpoint history or metric_value.')

        print(f'Resumed from checkpoint: {resume_checkpoint}')
        print(f'Starting at epoch {start_epoch}; previous best metric: {previous_best_metric}')

        return history, start_epoch, previous_best_metric 
