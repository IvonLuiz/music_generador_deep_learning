from typing import Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.torch.pixel_cnn import ConditionalGatedPixelCNN


class HierarchicalCondGatedPixelCNN(nn.Module):
    """
    Hierarchical Conditional Gated PixelCNN for modeling discrete latent variables
    in a hierarchical VQ-VAE setup.

    This implementation is based on the paper Generating Diverse High-Fidelity Images with VQ-VAE-2 (https://arxiv.org/abs/1906.00446).
    """
    def __init__(self,
                 num_prior_levels: int = 2,
                 input_size: Sequence[Tuple[int, int]] = ((32, 32), (64, 64)),
                 hidden_units: Sequence[int] = (512, 512),
                 residual_units: Sequence[int] = (2048, 1024),
                 num_layers: Sequence[int] = (20, 20),
                 num_classes: int = 256, # not used for now
                 attention_layers: Sequence[int] = (4, 0),
                 attention_heads: Sequence[Optional[int]] = (8, None),
                 conv_filter_size: Sequence[int] = (5, 5),
                 conditioning_stack_residual_blocks: Sequence[Optional[int]] = (None, 20),
                 dropout: Sequence[float] = (0.1, 0.1),
                 num_embeddings: Sequence[int] = (512, 512),
                 two_level_conditioning_mode: str = 'deconv') -> None:
        """
        Initialize the Conditional Gated PixelCNN model.
        @param num_prior_levels: Number of hierarchical prior levels (2 or 3)
        @param input_size: List of input sizes (H, W) for each prior level
        @param hidden_units: List of hidden units for each prior level
        @param residual_units: List of residual units for each prior level
        
        @param in_channels: Number of input channels (ignored if num_embeddings is provided)
        @param hidden_channels: Number of hidden channels
        @param num_layers: Number of Gated PixelCNN layers hidden
        @param kernel_size: Kernel size for convolutions
        @param conditional_dim: Dimension of the conditional vector/map
        @param spatial_conditioning: Whether to use spatial conditioning (2D map) or global (vector)
        @param num_classes: Number of output classes (e.g., 256 for 8-bit quantization)
        @param num_embeddings: Size of embedding dictionary (if input is discrete indices)
        """
        super().__init__()
        self.num_prior_levels = num_prior_levels
        self.two_level_conditioning_mode = two_level_conditioning_mode

        if two_level_conditioning_mode not in ('deconv', 'conv'):
            raise ValueError("two_level_conditioning_mode must be 'deconv' or 'conv'")

        if num_prior_levels <= 1 or num_prior_levels > 3:
            raise ValueError("num_prior_levels must be 2 or 3")

        for prior_levels in range(num_prior_levels):
            assert isinstance(input_size[prior_levels], tuple), "input_size must be a list of tuples"
            assert isinstance(hidden_units[prior_levels], int), "hidden_units must be a list of integers"
            assert isinstance(residual_units[prior_levels], int), "residual_units must be a list of integers"
            assert isinstance(num_layers[prior_levels], int), "num_layers must be a list of integers"
            assert isinstance(conv_filter_size[prior_levels], int), "conv_filter_size must be a list of integers"
            assert isinstance(num_embeddings[prior_levels], int), "num_embeddings must be a list of integers"

        # Prior levels acccording to num_prior_levels
        self.top_prior = ConditionalGatedPixelCNN(
            in_channels=1,
            hidden_channels=hidden_units[0],
            num_layers=num_layers[0],
            kernel_size=conv_filter_size[0],
            conditional_dim=None,
            num_classes=num_embeddings[0],
            num_embeddings=num_embeddings[0]
        )

        # Two-level setup: top -> bottom
        if num_prior_levels == 2:
            self.bottom_level = ConditionalGatedPixelCNN(
                in_channels=1,
                hidden_channels=hidden_units[-1],
                num_layers=num_layers[-1],
                kernel_size=conv_filter_size[-1],
                conditional_dim=hidden_units[-1],
                spatial_conditioning=True,
                num_classes=num_embeddings[-1],
                num_embeddings=num_embeddings[-1]
            )

            self.top_embedding_for_cond = nn.Embedding(num_embeddings[0], hidden_units[-1])
            if self.two_level_conditioning_mode == 'deconv':
                self.conditioning_stack = nn.Sequential(
                    nn.ConvTranspose2d(hidden_units[-1], hidden_units[-1], kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(hidden_units[-1], hidden_units[-1], kernel_size=3, padding=1),
                    nn.ReLU()
                )
            else:
                self.conditioning_stack = nn.Sequential(
                    nn.Conv2d(hidden_units[-1], hidden_units[-1], kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(hidden_units[-1], hidden_units[-1], kernel_size=3, padding=1),
                    nn.ReLU()
                )

        # Three-level setup: top -> middle -> bottom
        if num_prior_levels == 3:
            # Middle prior conditioned on top
            self.middle_level = ConditionalGatedPixelCNN(
                in_channels=1,
                hidden_channels=hidden_units[1],
                num_layers=num_layers[1],
                kernel_size=conv_filter_size[1],
                conditional_dim=hidden_units[1],
                spatial_conditioning=True,
                num_classes=num_embeddings[1],
                num_embeddings=num_embeddings[1]
            )

            self.top_embedding_for_middle_cond = nn.Embedding(num_embeddings[0], hidden_units[1])
            self.top_to_middle_conditioning = nn.Sequential(
                nn.Conv2d(hidden_units[1], hidden_units[1], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_units[1], hidden_units[1], kernel_size=3, padding=1),
                nn.ReLU(),
            )

            # Bottom prior conditioned on middle
            self.bottom_level = ConditionalGatedPixelCNN(
                in_channels=1,
                hidden_channels=hidden_units[2],
                num_layers=num_layers[2],
                kernel_size=conv_filter_size[2],
                conditional_dim=hidden_units[2],
                spatial_conditioning=True,
                num_classes=num_embeddings[2],
                num_embeddings=num_embeddings[2]
            )

            self.middle_embedding_for_bottom_cond = nn.Embedding(num_embeddings[1], hidden_units[2])
            self.middle_to_bottom_conditioning = nn.Sequential(
                nn.Conv2d(hidden_units[2], hidden_units[2], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_units[2], hidden_units[2], kernel_size=3, padding=1),
                nn.ReLU(),
            )

    def _prepare_indices(self, cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim == 4 and cond.shape[1] == 1:
            cond = cond.squeeze(1)
        return cond.long()

    def _build_spatial_condition(self,
                                 cond_indices: torch.Tensor,
                                 embedding_layer: nn.Embedding,
                                 conditioning_stack: nn.Module,
                                 target_spatial_shape: Tuple[int, int]) -> torch.Tensor:
        cond_emb = embedding_layer(cond_indices).permute(0, 3, 1, 2).contiguous()
        cond_map = conditioning_stack(cond_emb)
        if cond_map.shape[-2:] != target_spatial_shape:
            cond_map = F.interpolate(cond_map, size=target_spatial_shape, mode='nearest')
        return cond_map
        
    def forward(self, x, cond=None, level='top'):
        """
        Forward pass through the hierarchical PixelCNN.

        Args:
            x (torch.Tensor): Input tensor of shape [B, 1, H, W] containing discrete indices.
            cond (torch.Tensor, optional): Conditional tensor (indices) for lower levels.
            level (str): 'top', 'mid', or 'bottom' to specify which prior to use.

        Returns:
            torch.Tensor: Output logits of shape [B, num_classes, 1, H, W].
        """
        if level == 'top':
            return self.top_prior(x)

        elif level == 'mid':
            if self.num_prior_levels < 3 or not hasattr(self, 'middle_level'):
                raise ValueError("Mid level prior is not defined for num_prior_levels < 3")
            if cond is None:
                raise ValueError("Conditioning information (top level indices) required for mid level.")

            cond = self._prepare_indices(cond)
            cond_map = self._build_spatial_condition(
                cond_indices=cond,
                embedding_layer=self.top_embedding_for_middle_cond,
                conditioning_stack=self.top_to_middle_conditioning,
                target_spatial_shape=x.shape[-2:],
            )
            return self.middle_level(x, cond=cond_map)

        elif level == 'bottom':
            if not hasattr(self, 'bottom_level'):
                raise ValueError("Bottom level prior is not defined for num_prior_levels < 2")

            if cond is None:
                expected = "top" if self.num_prior_levels == 2 else "middle"
                raise ValueError(f"Conditioning information ({expected} level indices) required for bottom level.")

            cond = self._prepare_indices(cond)

            if self.num_prior_levels == 2:
                cond_map = self._build_spatial_condition(
                    cond_indices=cond,
                    embedding_layer=self.top_embedding_for_cond,
                    conditioning_stack=self.conditioning_stack,
                    target_spatial_shape=x.shape[-2:],
                )
            else:
                cond_map = self._build_spatial_condition(
                    cond_indices=cond,
                    embedding_layer=self.middle_embedding_for_bottom_cond,
                    conditioning_stack=self.middle_to_bottom_conditioning,
                    target_spatial_shape=x.shape[-2:],
                )

            return self.bottom_level(x, cond=cond_map)
        else:
            raise ValueError("level must be 'top', 'mid', or 'bottom'")

    def _sample_autoregressive(self,
                               shape,
                               cond=None,
                               level: str = 'top',
                               temperature: float = 1.0,
                               top_k: Optional[int] = None):
        """Naive autoregressive sampling by raster-scan over H and W."""
        device = next(self.parameters()).device
        if len(shape) != 4:
            raise ValueError(f"shape must be (B, 1, H, W), got {shape}")

        batch_size, channels, height, width = shape
        if channels != 1:
            raise ValueError(f"Expected channels==1 for discrete indices, got {channels}")

        # Start from an all-zero index grid
        x = torch.zeros((batch_size, 1, height, width), dtype=torch.long, device=device)

        for i in range(height):
            for j in range(width):
                # Forward pass through the chosen level
                logits = self.forward(x, cond=cond, level=level)  # (B, C, 1, H, W)
                logits_hw = logits.squeeze(2)[:, :, i, j]         # (B, C)

                # Temperature scaling
                if temperature <= 0:
                    raise ValueError(f"temperature must be > 0, got {temperature}")
                logits_hw = logits_hw / temperature

                # Top-k filtering
                if top_k is not None and top_k > 0 and top_k < logits_hw.shape[1]:
                    kth_vals = torch.topk(logits_hw, k=top_k, dim=1).values[:, -1].unsqueeze(1)
                    logits_hw = torch.where(
                        logits_hw < kth_vals,
                        torch.full_like(logits_hw, float('-inf')),
                        logits_hw,
                    )

                probs = torch.softmax(logits_hw, dim=1)           # (B, C)
                samples = torch.multinomial(probs, num_samples=1).squeeze(1)
                x[:, 0, i, j] = samples

        return x

    def generate(self,
                 shape,
                 cond=None,
                 level='top',
                 temperature: float = 1.0,
                 top_k: Optional[int] = None):
        """
        Generate samples from the hierarchical PixelCNN.

        Args:
            shape (tuple): Output shape (B, 1, H, W).
            cond (torch.Tensor, optional): Conditioning indices or maps for lower levels.
            level (str): 'top', 'mid', or 'bottom' to specify which prior to use.
        """
        if level == 'top':
            return self._sample_autoregressive(shape, cond=None, level='top', temperature=temperature, top_k=top_k)
        elif level == 'mid':
            if self.num_prior_levels < 3 or not hasattr(self, 'middle_level'):
                raise ValueError("Mid level prior is not defined for num_prior_levels < 3")
            return self._sample_autoregressive(shape, cond=cond, level='mid', temperature=temperature, top_k=top_k)
        elif level == 'bottom':
            if not hasattr(self, 'bottom_level'):
                raise ValueError("Bottom level prior is not defined for num_prior_levels < 2")
            if cond is None:
                raise ValueError("Conditioning information (top level indices) required for bottom level generation.")
            return self._sample_autoregressive(shape, cond=cond, level='bottom', temperature=temperature, top_k=top_k)
        else:
            raise ValueError("level must be 'top', 'mid', or 'bottom'")