from typing import Tuple, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual_stack import ResidualStack


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_layers, stride, kernel_size=4, padding=1, conv_type=2, num_downsample_blocks=1):
        """!
        @in_channels: Number of input channels
        @out_channels: Number of output channels
        @num_residual_layers: Number of residual layers in the Residual Stack
        @stride: Stride for the upsampling convolution. Can be an int or a tuple (stride_h, stride_w)
        @kernel_size: Kernel size for the upsampling convolution
        @padding: Padding for the upsampling convolution
        @param conv_type: Convolution type. 1 for standard conv, 2 for [kernel_size X kernel_size] conv with stride 2 to reduce checkerboard artifacts
        @param num_downsample_blocks: Number of downsampling blocks to apply before the residual stack (for deeper decoders)
        """
        super().__init__()
        self.layers = []
        if conv_type == 2:
            stride = (stride, stride) if isinstance(stride, int) else stride

        # Residual Stack before upsampling
        self.layers.append(ResidualStack(in_channels=in_channels,
                                    num_hiddens=in_channels,
                                    num_residual_hiddens=in_channels // 2,
                                    num_residual_layers=num_residual_layers,
                                    conv_type=conv_type))

        # Upsampling conv (Transpose Conv)
        self._create_decoding_layer(conv_type, in_channels, out_channels, stride, padding, kernel_size, num_downsample_blocks)

        self.layers.append(nn.ReLU(inplace=True))

        self._net = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self._net(x)
        return x

    def _create_decoding_layer(self, conv_type, in_channels, out_channels, stride, padding, kernel_size=4, num_downsample_blocks=1):
        """Create a 2D or 1D decoder block with the given parameters."""
        if conv_type == 2:
            conv_block = nn.ConvTranspose2d
            batch_norm = nn.BatchNorm2d
        elif conv_type == 1:
            conv_block = nn.ConvTranspose1d
            batch_norm = nn.BatchNorm1d
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")


        current_in = in_channels
        for block_idx in range(num_downsample_blocks):
            current_out = out_channels if block_idx == num_downsample_blocks - 1 else current_in
            self.layers.append(conv_block(current_in,
                                         current_out,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding))
            self.layers.append(batch_norm(current_out))
            current_in = current_out
