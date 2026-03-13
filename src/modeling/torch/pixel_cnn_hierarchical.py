from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.torch.pixel_cnn import ConditionalGatedPixelCNN


class VQVAE2HierarchicalPixelCNN(nn.Module):
    """Hierarchical PixelCNN for standard VQ-VAE2-style top/bottom priors."""

    def __init__(
        self,
        num_prior_levels: int = 2,
        input_size=None,
        hidden_units=None,
        num_layers=None,
        conv_filter_size=None,
        dropout=None,
        num_embeddings=None,
        two_level_conditioning_mode: str = 'deconv',
        **kwargs,
    ) -> None:
        super().__init__()

        if two_level_conditioning_mode not in ('deconv', 'conv'):
            raise ValueError("two_level_conditioning_mode must be 'deconv' or 'conv'")

        self.two_level_conditioning_mode = two_level_conditioning_mode
        self.num_prior_levels = int(num_prior_levels)
        if self.num_prior_levels != 2:
            raise ValueError('VQVAE2HierarchicalPixelCNN only supports num_prior_levels=2 (top, bottom)')

        self.input_size = input_size if input_size is not None else [(32, 32), (64, 64)]
        self.hidden_units = hidden_units if hidden_units is not None else [512, 512]
        self.num_layers_legacy = num_layers if num_layers is not None else [20, 20]
        self.conv_filter_size_legacy = conv_filter_size if conv_filter_size is not None else [5, 5]
        self.num_embeddings_legacy = num_embeddings if num_embeddings is not None else [512, 512]

        self.top_prior = ConditionalGatedPixelCNN(
            in_channels=1,
            hidden_channels=self.hidden_units[0],
            num_layers=self.num_layers_legacy[0],
            kernel_size=self.conv_filter_size_legacy[0],
            conditional_dim=None,
            num_classes=self.num_embeddings_legacy[0],
            num_embeddings=self.num_embeddings_legacy[0],
        )

        self.bottom_level = ConditionalGatedPixelCNN(
            in_channels=1,
            hidden_channels=self.hidden_units[-1],
            num_layers=self.num_layers_legacy[-1],
            kernel_size=self.conv_filter_size_legacy[-1],
            conditional_dim=self.hidden_units[-1],
            spatial_conditioning=True,
            num_classes=self.num_embeddings_legacy[-1],
            num_embeddings=self.num_embeddings_legacy[-1],
        )
        self.top_embedding_for_cond = nn.Embedding(self.num_embeddings_legacy[0], self.hidden_units[-1])
        if self.two_level_conditioning_mode == 'deconv':
            self.conditioning_stack = nn.Sequential(
                nn.ConvTranspose2d(self.hidden_units[-1], self.hidden_units[-1], kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.hidden_units[-1], self.hidden_units[-1], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.conditioning_stack = nn.Sequential(
                nn.Conv2d(self.hidden_units[-1], self.hidden_units[-1], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.hidden_units[-1], self.hidden_units[-1], kernel_size=3, padding=1),
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

    def forward(self, x, cond=None, level='top'):
        if level == 'top':
            return self.top_prior(x)

        if level != 'bottom':
            raise ValueError("level must be 'top' or 'bottom' for VQ-VAE2 hierarchical prior")
        if cond is None:
            raise ValueError('Conditioning information (top level indices) required for bottom level.')

        cond = self._prepare_indices(cond)
        cond_map = self._build_spatial_condition(
            cond_indices=cond,
            embedding_layer=self.top_embedding_for_cond,
            conditioning_stack=self.conditioning_stack,
            target_spatial_shape=x.shape[-2:],
        )
        return self.bottom_level(x, cond=cond_map)

    def _sample_autoregressive(self, shape, cond=None, level='top', temperature: float = 1.0, top_k: Optional[int] = None):
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
            level = 'top'
        return self._sample_autoregressive(shape, cond=cond, level=level, temperature=temperature, top_k=top_k)


# Backward-compatible alias used by existing VQ-VAE2 scripts.
class HierarchicalCondGatedPixelCNN(VQVAE2HierarchicalPixelCNN):
    pass
