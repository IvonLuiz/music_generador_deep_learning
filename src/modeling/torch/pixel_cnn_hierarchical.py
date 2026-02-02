import torch
import torch.nn as nn

from modeling.torch.pixel_cnn import ConditionalGatedPixelCNN


class HierarchicalCondGatedPixelCNN(nn.Module):
    """
    Hierarchical Conditional Gated PixelCNN for modeling discrete latent variables
    in a hierarchical VQ-VAE setup.

    This implementation is based on the paper Generating Diverse High-Fidelity Images with VQ-VAE-2 (https://arxiv.org/abs/1906.00446).
    """
    def __init__(self,
                 num_prior_levels: int = 2,
                 input_size: int = [(32, 32), (64, 64)],
                 hidden_units: int = [512, 512],
                 residual_units: int = [2048, 1024],
                 num_layers: int = [20, 20],
                 num_classes: int = 256,
                 attention_layers: int = [4, 0],
                 attention_heads: int = [8, None],
                 conv_filter_size: int = [5, 5],
                 conditioning_stack_residual_blocks: int = [None, 20],
                 dropout: float = [0.1, 0.1],
                 num_embeddings: int = [512, 512]) -> None:
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

        if num_prior_levels <= 1 or num_prior_levels > 3:
            raise ValueError("num_prior_levels must be 2 or 3")

        for prior_levels in range(num_prior_levels):
            assert isinstance(input_size[prior_levels], tuple), "input_size must be a list of tuples"
            assert isinstance(hidden_units[prior_levels], int), "hidden_units must be a list of integers"
            assert isinstance(residual_units[prior_levels], int), "residual_units must be a list of integers"
            assert isinstance(num_layers[prior_levels], int), "num_layers must be a list of integers"
            assert isinstance(attention_layers[prior_levels], int), "attention_layers must be a list of integers"
            assert (attention_heads[prior_levels] is None) or isinstance(attention_heads[prior_levels], int), "attention_heads must be a list of integers or None"
            assert isinstance(conv_filter_size[prior_levels], int), "conv_filter_size must be a list of integers"
            assert (conditioning_stack_residual_blocks[prior_levels] is None) or isinstance(conditioning_stack_residual_blocks[prior_levels], int), "conditioning_stack_residual_blocks must be a list of integers or None"
            assert isinstance(dropout[prior_levels], float), "dropout must be a list of floats"
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
        
        # Bottom level prior PixelCNN conditioned on top level latents
        ## the values would always be the last in the list
        if num_prior_levels >= 2:
            self.bottom_level = ConditionalGatedPixelCNN(
                in_channels=1,
                hidden_channels=hidden_units[-1],
                num_layers=num_layers[-1],
                kernel_size=conv_filter_size[-1],
                conditional_dim=128, # Assuming conditioning stack outputs 128 channels (hidden dim of prior)
                spatial_conditioning=True,
                num_classes=num_embeddings[-1],
                num_embeddings=num_embeddings[-1]
            )
            
            # Conditioning stack (Upsampler) for Top -> Bottom
            # Assumes 2x upsampling needed (e.g. 32x32 -> 64x64)
            # Input: Top indices (embedded) (N, hidden, H, W)
            self.conditioning_stack = nn.Sequential(
                nn.ConvTranspose2d(hidden_units[-1], 128, kernel_size=4, stride=2, padding=1), # Emb size 256 -> 128
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.top_embedding_for_cond = nn.Embedding(num_embeddings[0], hidden_units[-1]) # Embed indices before upsampling

        
        # Middle level prior PixelCNN conditioned on top level latents
        ## only if 3 levels are used, and uses the middle values in the list
        if num_prior_levels == 3:
            self.mid_prior = ConditionalGatedPixelCNN(
                in_channels=1,
                hidden_channels=hidden_units[1],
                num_layers=num_layers[1],
                kernel_size=conv_filter_size[1],
                conditional_dim=hidden_units[1],
                spatial_conditioning=True,
                num_classes=num_classes,
                num_embeddings=num_embeddings[1]
            )
        
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
            if not hasattr(self, 'mid_prior'):
                raise ValueError("Mid level prior is not defined for num_prior_levels < 3")
            return self.mid_prior(x, cond)
        elif level == 'bottom':
            if not hasattr(self, 'bottom_level'):
                raise ValueError("Bottom level prior is not defined for num_prior_levels < 2")
            
            # Process conditioning information (Top Level Indices)
            if cond is None:
                raise ValueError("Conditioning information (top level indices) required for bottom level.")
                
            # cond is expected to be [B, 1, H_top, W_top] or [B, H_top, W_top]
            if cond.ndim == 4 and cond.shape[1] == 1:
                cond = cond.squeeze(1)
            
            cond = cond.long() # [B, H_top, W_top]
            cond_emb = self.top_embedding_for_cond(cond).permute(0, 3, 1, 2) # [B, hidden_units[-1], H_top, W_top]
            
            # Upsample to bottom resolution for conditioning
            cond_map = self.conditioning_stack(cond_emb) # [B, 128, H_bottom, W_bottom]
            
            return self.bottom_level(x, cond=cond_map)
        else:
            raise ValueError("level must be 'top', 'mid', or 'bottom'")

    def _sample_autoregressive(self, shape, cond=None, level: str = 'top'):
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
                probs = torch.softmax(logits_hw, dim=1)           # (B, C)
                samples = torch.multinomial(probs, num_samples=1).squeeze(1)
                x[:, 0, i, j] = samples

        return x

    def generate(self, shape, cond=None, level='top'):
        """
        Generate samples from the hierarchical PixelCNN.

        Args:
            shape (tuple): Output shape (B, 1, H, W).
            cond (torch.Tensor, optional): Conditioning indices or maps for lower levels.
            level (str): 'top', 'mid', or 'bottom' to specify which prior to use.
        """
        if level == 'top':
            return self._sample_autoregressive(shape, cond=None, level='top')
        elif level == 'mid':
            if not hasattr(self, 'mid_prior'):
                raise ValueError("Mid level prior is not defined for num_prior_levels < 3")
            return self._sample_autoregressive(shape, cond=cond, level='mid')
        elif level == 'bottom':
            if not hasattr(self, 'bottom_level'):
                raise ValueError("Bottom level prior is not defined for num_prior_levels < 2")
            if cond is None:
                raise ValueError("Conditioning information (top level indices) required for bottom level generation.")
            return self._sample_autoregressive(shape, cond=cond, level='bottom')
        else:
            raise ValueError("level must be 'top', 'mid', or 'bottom'")