from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.torch.pixel_cnn import ConditionalGatedPixelCNN


class JukeboxLevelPixelCNN(nn.Module):
    """Single-prior PixelCNN for Jukebox top/middle/bottom training."""

    def __init__(
        self,
        level: int,
        hidden_channels: int,
        num_layers: int,
        conv_filter_size: int,
        num_embeddings: int,
        cond_num_embeddings: Optional[int] = None,
        two_level_conditioning_mode: str = 'deconv',
    ) -> None:
        super().__init__()

        if two_level_conditioning_mode not in ('deconv', 'conv'):
            raise ValueError("two_level_conditioning_mode must be 'deconv' or 'conv'")
        if not isinstance(level, int) or level < 1 or level > 3:
            raise ValueError('level must be an integer in [1, 2, 3]')
        if not isinstance(num_embeddings, int) or num_embeddings <= 1:
            raise ValueError('num_embeddings must be an integer > 1')

        self.level = level
        self.is_conditional = level in (2, 3)
        self.two_level_conditioning_mode = two_level_conditioning_mode

        conditional_dim = hidden_channels if self.is_conditional else None
        self.pixelcnn_prior = ConditionalGatedPixelCNN(
            in_channels=1,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=conv_filter_size,
            conditional_dim=conditional_dim,
            spatial_conditioning=self.is_conditional,
            num_classes=num_embeddings,
            num_embeddings=num_embeddings,
        )

        self.cond_embedding: Optional[nn.Embedding] = None
        self.conditioning_stack: Optional[nn.Module] = None

        if self.is_conditional:
            if cond_num_embeddings is None:
                raise ValueError('cond_num_embeddings must be provided for level 2/3 priors')
            self.cond_embedding = nn.Embedding(cond_num_embeddings, hidden_channels)
            if self.two_level_conditioning_mode == 'deconv':
                self.conditioning_stack = nn.Sequential(
                    nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
            else:
                self.conditioning_stack = nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                )

    @staticmethod
    def _prepare_indices(cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim == 4 and cond.shape[1] == 1:
            cond = cond.squeeze(1)
        return cond.long()

    @staticmethod
    def _build_spatial_condition(
        cond_indices: torch.Tensor,
        embedding_layer: nn.Embedding,
        conditioning_stack: nn.Module,
        target_spatial_shape: Tuple[int, int],
    ) -> torch.Tensor:
        cond_emb = embedding_layer(cond_indices).permute(0, 3, 1, 2).contiguous()
        cond_map = conditioning_stack(cond_emb)
        if cond_map.shape[-2:] != target_spatial_shape:
            cond_map = F.interpolate(cond_map, size=target_spatial_shape, mode='nearest')
        return cond_map

    def _level_to_int(self, level) -> int:
        if level is None:
            return -1
        if isinstance(level, int):
            return level
        mapping = {'top': 1, 'mid': 2, 'middle': 2, 'bottom': 3}
        if level not in mapping:
            raise ValueError("level must be one of: top, mid/middle, bottom")
        return mapping[level]

    def forward(self, x, cond=None, level=None):
        if level is not None:
            expected_level = self._level_to_int(level)
            if expected_level != self.level:
                raise ValueError(f'This model is instantiated for level={self.level}, but received level={level}')

        if not self.is_conditional:
            return self.pixelcnn_prior(x)

        if cond is None:
            cond_name = 'top' if self.level == 2 else 'middle'
            raise ValueError(f'Conditioning information ({cond_name} indices) is required for level={self.level}')

        if self.cond_embedding is None or self.conditioning_stack is None:
            raise RuntimeError('Conditional model missing cond_embedding/conditioning_stack')

        cond = self._prepare_indices(cond)
        cond_map = self._build_spatial_condition(
            cond_indices=cond,
            embedding_layer=self.cond_embedding,
            conditioning_stack=self.conditioning_stack,
            target_spatial_shape=x.shape[-2:],
        )
        return self.pixelcnn_prior(x, cond=cond_map)

    def _sample_autoregressive(self, shape, cond=None, level=None, temperature: float = 1.0, top_k: Optional[int] = None):
        device = next(self.parameters()).device
        if len(shape) != 4:
            raise ValueError(f'shape must be (B, 1, H, W), got {shape}')

        batch_size, channels, height, width = shape
        if channels != 1:
            raise ValueError(f'Expected channels==1 for discrete indices, got {channels}')

        x = torch.zeros((batch_size, 1, height, width), dtype=torch.long, device=device)

        for i in range(height):
            for j in range(width):
                logits = self.forward(x, cond=cond, level=level)
                logits_hw = logits.squeeze(2)[:, :, i, j]

                if temperature <= 0:
                    raise ValueError(f'temperature must be > 0, got {temperature}')
                logits_hw = logits_hw / temperature

                if top_k is not None and top_k > 0 and top_k < logits_hw.shape[1]:
                    kth_vals = torch.topk(logits_hw, k=top_k, dim=1).values[:, -1].unsqueeze(1)
                    logits_hw = torch.where(logits_hw < kth_vals, torch.full_like(logits_hw, float('-inf')), logits_hw)

                probs = torch.softmax(logits_hw, dim=1)
                samples = torch.multinomial(probs, num_samples=1).squeeze(1)
                x[:, 0, i, j] = samples

        return x

    def generate(self, shape, cond=None, level=None, temperature: float = 1.0, top_k: Optional[int] = None):
        if level is None:
            level = self.level
        expected_level = self._level_to_int(level)
        if expected_level != self.level:
            raise ValueError(f'This model is instantiated for level={self.level}, but generate received level={level}')
        return self._sample_autoregressive(shape, cond=cond, level=level, temperature=temperature, top_k=top_k)
