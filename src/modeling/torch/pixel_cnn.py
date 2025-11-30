import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple


class MaskedConv2d(nn.Conv2d):
    """
    A 2D convolutional layer with a mask applied to the weights.
    The mask ensures that the convolution respects the autoregressive property.
    """
    def __init__(self, mask_type: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert mask_type in ['A', 'B'], "mask_type must be either 'A' or 'B'"

        self.mask_type = mask_type
        self.register_buffer('mask', self.weight.data.clone())
        self.mask = self.create_mask()

    def create_mask(self) -> torch.Tensor:
        """
        Create mask for the convolutional layer.
        Type A masks out the center pixel, Type B includes it.
        @return: mask tensor of shape (1, 1, k_h, k_w)
        """
        _, _, kernel_height, kernel_width = self.weight.size()
        mask = torch.ones((kernel_height, kernel_width), dtype=torch.float32)
        center_height = kernel_height // 2
        center_width = kernel_width // 2
        
        # Mask out future pixels
        # For type A, also mask out the center pixel
        mask[center_height, center_width + (self.mask_type == 'B'):] = 0 # Mask out pixels to the right of center
        mask[center_height + 1:] = 0 # Mask out rows below the center row

        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


class GatedPixelCNNBlock(nn.Module):
    """
    A single Gated PixelCNN block with masked convolutions.
    """
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 mask_type: str = 'B') -> None:
        super().__init__()
        # Vertical stack for pixels above
        self.conv_vertical = MaskedConv2d(
            mask_type,
            in_channels,
            2 * out_channels,
            kernel_size,
            padding=kernel_size // 2
        )
        # Horizontal stack for pixels to the left
        self.conv_horizontal = MaskedConv2d(
            mask_type,
            in_channels,
            2 * out_channels,
            kernel_size,
            padding=kernel_size // 2
        )
        # Vertical and horizontal projections (1x1 convolutions)
        self.proj_vertical = nn.Conv2d(2 * out_channels, 2 * out_channels, kernel_size=1)
        self.proj_horizontal = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
        self.out_channels = out_channels

    def gated_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gated activation function.
        @param x: Input tensor of shape [B, 2*Channels, H, W]
        @return: Activated tensor of shape [B, Channels, H, W]
        """
        # Divide the channels into 2 halves (features and gates)
        tanh_out, sigmoid_out = x.chunk(2, dim=1)
        # tanh(f) * sigmoid(g)
        return torch.tanh(tanh_out) * torch.sigmoid(sigmoid_out)

    def forward(self, x: torch.Tensor, h_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Gated PixelCNN block.
        Formula: $$y = \tanh(W_{f} * x) \odot \sigma(W_{g} * x)$$
        @param x: Input tensor of shape [B, In_Channels, H, W]
        @param h_input: Conditional input tensor of shape [B, In_Channels, H, W]
        @return: Output tensor of shape [B, Out_Channels, H, W]
        """
        # Vertical and horizontal convolutions
        v_val = self.conv_vertical(x)               # [B, 2*Out, H, W]
        h_val = self.conv_horizontal(x)             # [B, 2*Out, H, W]
        
        # Connection from vertical to horizontal
        v_proj = self.proj_vertical(v_val)          # [B, 2*Out, H, W]
        h_val = h_val + v_proj                      # [B, 2*Out, H, W]
        
        # Gated activation
        v_val = self.gated_activation(v_val)        # [B, Out, H, W]
        h_val = self.gated_activation(h_val)        # [B, Out, H, W]
        
        # Residual connection for horizontal stack
        if self.mask_type == 'B':
            h_proj = self.proj_horizontal(h_val)    # [B, Out, H, W]
        
        return v_proj + h_proj


class ConditionalGatedPixelCNN(nn.Module):
    """
    Gated PixelCNN model with conditional input.
    """
    def __init__(self,
                 in_channels: int = 1,
                 hidden_channels: int = 64,
                 num_layers: int = 5,
                 kernel_size: int = 3) -> None:
        """
        Initialize the Conditional Gated PixelCNN model.
        @param in_channels: Number of input channels
        @param hidden_channels: Number of hidden channels
        @param num_layers: Number of Gated PixelCNN layers hidden
        @param kernel_size: Kernel size for convolutions
        """
        super().__init__()

        # Mask type A for the first layer
        self.input_conv = GatedPixelCNNBlock('A', in_channels, hidden_channels, kernel_size)
        # Mask type B for subsequent layers
        self.gated_blocks = nn.ModuleList([
            GatedPixelCNNBlock('B', hidden_channels, hidden_channels, kernel_size)
            for _ in range(num_layers)
        ])

        # Output layers with ReLu and 1x1 convs
        self.output_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        @param x: Input of normalized tensor of shape [B, C, H, W]
        @param cond: Conditional tensor of shape [B, C, H, W]
        @return: Output tensor of shape [B, C, H, W]
        """
        x = x.float()
        x = self.input_conv(x)
        cond = self.cond_conv(cond)
        x = x + cond
        
        for block in self.gated_blocks:
            x = block(x) + x  # Residual connection
        
        x = self.output_conv(x)
        return x

