from typing import Tuple, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual_stack import ResidualStack


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_layers, stride, kernel_size=4, padding=1, conv_type=2):
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
        if conv_type == 1:
            # standard convolution (used for raw audio)
            self._create_2d_encoder(in_channels, out_channels, num_residual_layers, stride, padding)
        elif conv_type == 2:
            # [kernel_size X kernel_size] conv with stride 2 to reduce checkerboard artifacts (used for spectrograms)
            self._create_2d_encoder(in_channels, out_channels, num_residual_layers, stride, padding)
        else:
            raise ValueError(f"Invalid conv_type: {conv_type}. Must be 1 or 2.")

        self.layers.append(nn.ReLU(inplace=True))

        # Residual Stack after downsampling
        self.layers.append(ResidualStack(in_channels=out_channels,
                                    num_hiddens=out_channels,
                                    num_residual_hiddens=out_channels // 2,
                                    num_residual_layers=num_residual_layers))

        self._net = nn.Sequential(*self.layers)
    
    def forward(self, x):
        x = self._net(x)
        return x

    def _create_2d_encoder(self, in_channels, out_channels, num_residual_layers, stride, padding, kernel_size=4):
        """Create a 2D encoder block with the given parameters."""
        self.layers.append(nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding))
        self.layers.append(nn.BatchNorm2d(out_channels))

    def _create_1d_encoder(self, in_channels, out_channels, num_residual_layers, stride, padding, kernel_size=4):
        """Create a 1D encoder block with the given parameters."""
        self.layers.append(nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding))
        self.layers.append(nn.BatchNorm1d(out_channels))
