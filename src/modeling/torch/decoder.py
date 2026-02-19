from typing import Tuple, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual_stack import ResidualStack


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_layers, stride, kernel_size=4, padding=1, conv_type=2):
        """!
        @in_channels: Number of input channels
        @out_channels: Number of output channels
        @num_residual_layers: Number of residual layers in the Residual Stack
        @stride: Stride for the upsampling convolution. Can be an int or a tuple (stride_h, stride_w)
        @kernel_size: Kernel size for the upsampling convolution
        @padding: Padding for the upsampling convolution
        @param conv_type: Convolution type. 1 for standard conv, 2 for [kernel_size X kernel_size] conv with stride 2 to reduce checkerboard artifacts
        """
        super().__init__()
        self.layers = []
        stride = (stride, stride) if isinstance(stride, int) else stride  # normalize stride to tuple if it's an int

        # Residual Stack before upsampling
        self.layers.append(ResidualStack(in_channels=in_channels,
                                    num_hiddens=in_channels,
                                    num_residual_hiddens=in_channels // 2,
                                    num_residual_layers=num_residual_layers))

        # Upsampling conv (Transpose Conv)
        if conv_type == 1:
            # standard transpose convolution (used for raw audio)
            self._create_1d_decoder(in_channels, out_channels, num_residual_layers, stride, padding, kernel_size)
        elif conv_type == 2:
            # [kernel_size X kernel_size] transpose conv with stride 2 (used for spectrograms)
            self._create_2d_decoder(in_channels, out_channels, num_residual_layers, stride, padding, kernel_size)

        self.layers.append(nn.ReLU(inplace=True))

        self._net = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self._net(x)
        return x

    def _create_2d_decoder(self, in_channels, out_channels, num_residual_layers, stride, padding, kernel_size=4):
        """Create a 2D decoder block with the given parameters."""
        self.layers.append(nn.ConvTranspose2d(in_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding))
        self.layers.append(nn.BatchNorm2d(out_channels))

    def _create_1d_decoder(self, in_channels, out_channels, num_residual_layers, stride, padding, kernel_size=4):
        """Create a 1D decoder block with the given parameters."""
        self.layers.append(nn.ConvTranspose1d(in_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding))
        self.layers.append(nn.BatchNorm1d(out_channels))