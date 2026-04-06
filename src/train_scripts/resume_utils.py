import os
import json
import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional

def extract_best_metric(history: dict, val_key: str = 'val_loss', train_key: str = 'train_loss') -> Optional[float]:
    if not history:
        return None
    if val_key in history and history[val_key]:
        finite_vals = [float(v) for v in history[val_key] if np.isfinite(v)]
        if finite_vals:
            return min(finite_vals)
    if train_key in history and history[train_key]:
        finite_vals = [float(v) for v in history[train_key] if np.isfinite(v)]
        if finite_vals:
            return min(finite_vals)
    return None

def load_resume_artifacts(pretrained_weights_path: str, val_key: str = 'val_loss', train_key: str = 'train_loss') -> Tuple[dict, Optional[float], Any]:
    if not os.path.isfile(pretrained_weights_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_weights_path}")

    checkpoint = torch.load(pretrained_weights_path, map_location='cpu', weights_only=False)
    resume_history = checkpoint.get('history', {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(resume_history, dict):
        resume_history = {}

    history_path = os.path.join(os.path.dirname(pretrained_weights_path), 'loss_history.json')
    if os.path.isfile(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            file_history = json.load(f)
        if isinstance(file_history, dict):
            resume_history = file_history

    best_metric = extract_best_metric(resume_history, val_key, train_key)
    
    if best_metric is None and isinstance(checkpoint, dict):
        if val_key in checkpoint:
            best_metric = float(checkpoint[val_key])
        elif train_key in checkpoint:
            best_metric = float(checkpoint[train_key])
        elif 'metric_value' in checkpoint:
            best_metric = float(checkpoint['metric_value'])
        elif 'loss' in checkpoint:
            best_metric = float(checkpoint['loss'])

    return resume_history, best_metric, checkpoint

import json
import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional

def extract_best_metric(history: dict, val_key: str = 'val_loss', train_key: str = 'train_loss') -> Optional[float]:
    if not history:
        return None
    if val_key in history and history[val_key]:
        finite_vals = [float(v) for v in history[val_key] if np.isfinite(v)]
        if finite_vals:
            return min(finite_vals)
    if train_key in history and history[train_key]:
        finite_vals = [float(v) for v in history[train_key] if np.isfinite(v)]
        if finite_vals:
            return min(finite_vals)
    return None

def load_resume_artifacts(pretrained_weights_path: str, val_key: str = 'val_loss', train_key: str = 'train_loss') -> Tuple[dict, Optional[float], Any]:
    if not os.path.isfile(pretrained_weights_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_weights_path}")

    checkpoint = torch.load(pretrained_weights_path, map_location='cpu', weights_only=False)
    resume_history = checkpoint.get('history', {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(resume_history, dict):
        resume_history = {}

    history_path = os.path.join(os.path.dirname(pretrained_weights_path), 'loss_history.json')
    if os.path.isfile(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            file_history = json.load(f)
        if isinstance(file_history, dict):
            resume_history = file_history

    best_metric = extract_best_metric(resume_history, val_key, train_key)
    
    if best_metric is None and isinstance(checkpoint, dict):
        if val_key in checkpoint:
            best_metric = float(checkpoint[val_key])
        elif train_key in checkpoint:
            best_metric = float(checkpoint[train_key])
        elif 'metric_value' in checkpoint:
            best_metric = float(checkpoint['metric_value'])
        elif 'loss' in checkpoint:
            best_metric = float(checkpoint['loss'])

    return resume_history, best_metric, checkpoint
