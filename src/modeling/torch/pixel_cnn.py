import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple


class PixelConvLayer(nn.Module):
    """
    A single layer of PixelCNN with masked convolutions.
    """
    def __init__(self, mask_type: str, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.mask_type = mask_type
        self.register_buffer('mask', self.create_mask())
    
    def create_mask(self) -> torch.Tensor:
        """
        Create mask for the convolutional layer.
        Type A masks out the center pixel, Type B includes it.
        @return: mask tensor of shape (1, 1, k, k)
        """
        k = self.conv.kernel_size[0]
        mask = torch.ones((k, k), dtype=torch.float32)
        center = k // 2
        mask[center, center + (self.mask_type == 'B'):] = 0
        mask[center + 1:] = 0
        return mask.unsqueeze(0).unsqueeze(0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.conv.weight.data *= self.mask
        return self.conv(x)

class PixelCNN(nn.Module):
    """
    PixelCNN model for autoregressive modeling of images.
    """
    def __init__(self, in_channels: int, num_layers: int, hidden_channels: int, kernel_size: int) -> None:
        super().__init__()
        layers = []
        
        # First layer is type A
        layers.append(PixelConvLayer('A', in_channels, hidden_channels, kernel_size))
        layers.append(nn.ReLU(inplace=True))
        
        # Subsequent layers are type B
        for _ in range(num_layers - 1):
            layers.append(PixelConvLayer('B', hidden_channels, hidden_channels, kernel_size))
            layers.append(nn.ReLU(inplace=True))
        
        # Final layer to output logits for each pixel value
        layers.append(nn.Conv2d(hidden_channels, in_channels, kernel_size=1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)