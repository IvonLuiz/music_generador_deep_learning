from typing import Iterable, Tuple
import torch.nn as nn
import torch.nn.functional as F


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens, conv_type=2, dilation=1):
        super().__init__()
        if conv_type == 2:
            Conv = nn.Conv2d
            BatchNorm = nn.BatchNorm2d
        elif conv_type == 1:
            Conv = nn.Conv1d
            BatchNorm = nn.BatchNorm1d
        else:
            raise ValueError("conv_type must be either 1 (Conv1d) or 2 (Conv2d)")
        
        # Jukebox architecture for residual block:
        # 1. Conv (3x3 or 3), ReLU
        # 2. Conv (1x1 or 1), BatchNorm, ReLU
        # Note: Jukebox uses specific activations, but here we'll stick to standard ResNet structure 
        # adapted with dilation.
        
        # In Jukebox: "Each encoder block consists of a downsampling convolution, a residual network... 
        # Dilation is grown by a factor of 3 in these residual networks"
        
        self._block = nn.Sequential(
            nn.ReLU(inplace=False),
            Conv(in_channels=in_channels,
                 out_channels=num_residual_hiddens,
                 kernel_size=3, stride=1,
                 padding=dilation, # padding must be equal to dilation to maintain the same spatial dimensions
                 dilation=dilation,
                 bias=False),
            BatchNorm(num_residual_hiddens),
            nn.ReLU(inplace=False),
            Conv(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False),
            BatchNorm(num_hiddens)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens, num_residual_layers, dilation=1, dilation_growth_rate=1, conv_type=2):
        super().__init__()
        self._num_residual_layers = num_residual_layers
        layers = []

        current_dilation = 1
        for _ in range(self._num_residual_layers):
            layers.append(ResidualLayer(in_channels, num_hiddens, num_residual_hiddens,
                                        dilation=current_dilation, conv_type=conv_type))
            if dilation_growth_rate > 1:
                current_dilation *= dilation_growth_rate  # grow dilation by a factor of 3 for each subsequent layer
        
        self._layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x
