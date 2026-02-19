from typing import Tuple, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual_stack import ResidualStack


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_layers, stride, kernel_size=4, padding=1, conv_type=2, num_downsample_blocks=1):
        """!
        @in_channels: Number of input channels
        @out_channels: Number of output channels
        @num_residual_layers: Number of residual layers in the Residual Stack
        @stride: Stride for the downsampling convolution. Can be an int or a tuple (stride_h, stride_w)
        @kernel_size: Kernel size for the downsampling convolution
        @padding: Padding for the downsampling convolution
        @param conv_type: Convolution type. 1 for standard conv, 2 for [kernel_size X kernel_size] conv with stride 2 to reduce checkerboard artifacts 
        """
        super().__init__()
        self.layers = []
        stride = (stride, stride) if isinstance(stride, int) else stride  # normalize stride to tuple if it's an int

        # Downsampling conv
        # conv_type can be 1 for raw audio or 2 for spectrograms
        self._create_encoder_layer(conv_type, in_channels, out_channels, stride, padding, kernel_size)

        self.layers.append(nn.ReLU(inplace=True))

        # Residual Stack after downsampling
        self.layers.append(ResidualStack(in_channels=out_channels,
                                        num_hiddens=out_channels,
                                        num_residual_hiddens=out_channels // 2,
                                        num_residual_layers=num_residual_layers,
                                        conv_type=conv_type))

        self._net = nn.Sequential(*self.layers)
    
    def forward(self, x):
        x = self._net(x)
        return x

    def _create_encoder_layer(self, conv_size, in_channels, out_channels, stride, padding, kernel_size=4):
        """Create a 2D or 1D encoder block with the given parameters."""
        if conv_size == 1:
            conv_type = nn.Conv1d
            batch_norm = nn.BatchNorm1d
        elif conv_size == 2:
            conv_type = nn.Conv2d
            batch_norm = nn.BatchNorm2d
        else:
            raise ValueError(f"Unsupported conv_size: {conv_size}")

        self.layers.append(conv_type(in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding))
        self.layers.append(batch_norm(out_channels))
