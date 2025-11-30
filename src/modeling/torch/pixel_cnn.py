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
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mask_type: str = 'B',
                 kernel_size: int = 3,
                 conditional_dim: int = None) -> None:
        super().__init__()
        self.mask_type = mask_type
        padding = kernel_size // 2

        # Vertical stack for pixels above
        self.conv_vertical = MaskedConv2d(
            mask_type,
            in_channels,
            2 * out_channels,
            kernel_size,
            padding = padding
        )
        # Horizontal stack for pixels to the left
        self.conv_horizontal = MaskedConv2d(
            mask_type,
            in_channels,
            2 * out_channels,
            kernel_size,
            padding = padding
        )
        # Vertical and horizontal projections (1x1 convolutions)
        self.proj_vertical = nn.Conv2d(
            2 * out_channels,
            2 * out_channels,
            kernel_size=1
        ) # connects vertical to horizontal
        self.proj_horizontal = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1
        ) # final convolution for horizontal residual connection
        
        # Conditioning
        self.conditional_dim = conditional_dim
        if conditional_dim is not None:
            # projects the vector h to sum before activations
            # W_f * h and W_g * h
            self.cond_proj_vertical = nn.Linear(
                conditional_dim, 2 * out_channels
            )
            self.cond_proj_horizontal = nn.Linear(
                conditional_dim, 2 * out_channels
            )

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

    def forward(self, vert_input: torch.Tensor, hor_input: torch.Tensor, cond = None) -> torch.Tensor:
        """
        Forward pass of the Gated PixelCNN block.
        Formula: $$y = \tanh(W_{f} * x) \odot \sigma(W_{g} * x)$$
        @param vert_input: Input tensor of shape [B, In_Channels, H, W]
        @param hor_input: Input tensor of shape [B, In_Channels, H, W]
        @param cond: Conditional input tensor of shape [B, Conditional_Dim]
        @return: Output tensor of shape [B, Out_Channels, H, W]
        """
        # Vertical and horizontal convolutions
        vert_val = self.conv_vertical(vert_input)       # [B, 2*Out, H, W]
        hor_val = self.conv_horizontal(hor_input)           # [B, 2*Out, H, W]
        
        # Connection from vertical to horizontal
        vert_proj = self.proj_vertical(vert_val)           # [B, 2*Out, H, W]
        hor_val = hor_val + vert_proj                          # [B, 2*Out, H, W]

        # Add conditioning if provided
        if self.conditional_dim is not None and cond is not None:
            # Project conditioning vector to match dimensions
            cond_vert = self.cond_proj_vertical(cond)      # [B, 2*Out]
            cond_horiz = self.cond_proj_horizontal(cond)   # [B, 2*Out]
            # Reshape for addition
            cond_vert = cond_vert.unsqueeze(2).unsqueeze(3)     # [B, 2*Out, 1, 1]
            cond_horiz = cond_horiz.unsqueeze(2).unsqueeze(3)   # [B, 2*Out, 1, 1]
            
            # Add conditioning (bias to certain features)
            vert_val = vert_val + cond_vert
            hor_val = hor_val + cond_horiz
        
        # Gated activation
        vert_out = self.gated_activation(vert_val)          # [B, Out, H, W]
        hor_out = self.gated_activation(hor_val)              # [B, Out, H, W]
        
        # Residual connection for horizontal stack
        if self.mask_type == 'B':
            hor_out = self.proj_horizontal(hor_out) + hor_input    # [B, Out, H, W]
        
        return vert_out, hor_out


class ConditionalGatedPixelCNN(nn.Module):
    """
    Gated PixelCNN model with conditional input.
    """
    def __init__(self,
                 in_channels: int = 1,
                 hidden_channels: int = 64,
                 num_layers: int = 5,
                 kernel_size: int = 3,
                 conditional_dim: int = None,
                 num_classes: int = 256) -> None:
        """
        Initialize the Conditional Gated PixelCNN model.
        @param in_channels: Number of input channels
        @param hidden_channels: Number of hidden channels
        @param num_layers: Number of Gated PixelCNN layers hidden
        @param kernel_size: Kernel size for convolutions
        @param conditional_dim: Dimension of the conditional vector
        @param num_classes: Number of output classes (e.g., 256 for 8-bit quantization)
        """
        super().__init__()
        self.num_classes = num_classes

        # Mask type A for the first layer
        self.input_conv = GatedPixelCNNBlock(in_channels, hidden_channels, 'A', kernel_size, conditional_dim)
        # Mask type B for subsequent layers
        self.gated_blocks = nn.ModuleList([
            GatedPixelCNNBlock(hidden_channels, hidden_channels, 'B', kernel_size, conditional_dim)
            for _ in range(num_layers)
        ])

        # Output layers with ReLu and 1x1 convs
        self.output_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels * num_classes, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        """
        @param x: Input of normalized tensor of shape [B, C, H, W]
        @param cond: Conditional tensor of shape [B, C, H, W]
        @return: Output tensor of shape [B, C, H, W]
        """
        x = x.float()
 
        # Initial vertical and horizontal stacks
        vert, hor = self.input_conv(x, x, cond) # receive same image
        
        # Pass through residual Gated PixelCNN blocks
        for block in self.gated_blocks:
            vert, hor = block(vert, hor, cond)

        # Post-process output (only horizontal stack is used for prediction)
        x = self.output_conv(hor) # relu -> conv2d -> relu -> conv2d

        # x is flattened
        # reshape for CrossEntropy: [B, 256, C, H, W] or [B, 256, H, W] if C=1
        batch, _, height, width = x.size()
        x = x.view(batch, self.num_classes, -1, height, width)  # [B, 256, C, H, W]

        return x

