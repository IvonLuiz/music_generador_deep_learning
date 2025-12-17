import torch
import torch.nn as nn
import torch.nn.functional as F

from vq_vae_residual import ResidualStack
from .vector_quantizer import VectorQuantizer

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_layers, stride, kernel_size=4, padding=1, dropout_rate=0.0):
        super().__init__()
        layers = []
        
        stride = (stride, stride) if isinstance(stride, int) else stride  # normalize stride to tuple if it's an int

        # Downsampling conv
        # 4x4 conv with stride 2 is standard for VQ-VAE to reduce checkerboard artifacts
        layers.append(nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        # Residual Stack after downsampling
        layers.append(ResidualStack(in_channels=out_channels,
                                    num_hiddens=out_channels,
                                    num_residual_hiddens=out_channels // 2,
                                    num_residual_layers=num_residual_layers))

        self._net = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self._net(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_layers, stride, kernel_size=4, padding=1, dropout_rate=0.0):
        super().__init__()
        layers = []

        # Residual Stack before upsampling
        layers.append(ResidualStack(in_channels=in_channels,
                                    num_hiddens=in_channels,
                                    num_residual_hiddens=in_channels // 2,
                                    num_residual_layers=num_residual_layers))

        stride = (stride, stride) if isinstance(stride, int) else stride  # normalize stride to tuple if it's an int

        # Upsampling conv (Transpose Conv)
        layers.append(nn.ConvTranspose2d(in_channels,
                                         out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        self._net = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self._net(x)
        return x

