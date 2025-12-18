from typing import Tuple, Iterable, List

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
        # 4x4 conv with stride 2  s standard for VQ-VAE to reduce checkerboard artifacts
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


class VQ_VAE_Hierarchical(nn.Module):
    """
    Hierarchical VQ-VAE model with two levels of encoders, decoders, and vector quantizers.
    
    This implementation on the paper Generating Diverse High-Fidelity Images with VQ-VAE-2 (https://arxiv.org/abs/1906.00446).
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 dim_bottom: int = 64,
                 dim_top: int = 128,
                 num_residual_layers: int = 2,
                 num_embeddings_top: int = 512,
                 num_embeddings_bottom: int = 512,
                 beta: float = 0.25,
                 dropout_rate: float = 0.0):
        """
        Args:
            dim_bottom: Channels for the bottom latent space.
            dim_top: Channels for the top latent space.
            embeddings_size: Size of the codebook (K).
        """
        
        super().__init__()
        H, W, C = input_shape
        
        ## ENCODER PATH
        # Bottom: image -> bottom latents (e.g., 256 -> 128 -> 64)
        self.encoder_bottom = nn.Sequential(
            EncoderBlock(in_channels=C,
                         out_channels=dim_bottom // 2,
                         num_residual_layers=num_residual_layers,
                         stride=2,
                         dropout_rate=dropout_rate), # H/2
            EncoderBlock(in_channels=dim_bottom // 2,
                         out_channels=dim_bottom,
                         num_residual_layers=num_residual_layers,
                         stride=2,
                         dropout_rate=dropout_rate) # H/4
        )
        
        # Top: bottom latents -> top latents (e.g., 64 -> 32)
        self.encoder_top = nn.Sequential(
            EncoderBlock(in_channels=dim_bottom,
                         out_channels=dim_top,
                         num_residual_layers=num_residual_layers,
                         stride=2,
                         dropout_rate=dropout_rate), # H/8
        )

        ## VECTOR QUANTIZERS
        self.vq_top = VectorQuantizer(num_embeddings=num_embeddings_top,
                                      embedding_dim=dim_top,
                                      beta=beta)
        self.vq_bottom = VectorQuantizer(num_embeddings=num_embeddings_bottom,
                                         embedding_dim=dim_bottom,
                                         beta=beta)
        
        ## PRE-QUANT CONVS
        # 1x1 conv to adjust channels before quantization if needed
        self.pre_vq_conv_top = nn.Conv2d(in_channels=dim_top,
                                         out_channels=dim_top,
                                         kernel_size=1)
        self.pre_vq_conv_bottom = nn.Conv2d(in_channels=dim_bottom,
                                            out_channels=dim_bottom,
                                            kernel_size=1)
        
        ## DECODER PATH
        # Top: top latents -> upsampled to bottom size
        self.decoder_top = nn.Sequential(
            DecoderBlock(in_channels=dim_top,
                         out_channels=dim_bottom,
                         stride=2,
                         num_residual_layers=num_residual_layers,
                         dropout_rate=dropout_rate), # H/4
        )
        
        # Bottom: bottom latents + upsampled top -> reconstructed image
        self.decoder_bottom = nn.Sequential(
            DecoderBlock(in_channels=dim_bottom * 2,
                         out_channels=dim_bottom // 2,
                         stride=2,
                         num_residual_layers=num_residual_layers,
                         dropout_rate=dropout_rate), # H/2
            DecoderBlock(in_channels=dim_bottom // 2,
                         out_channels=C,
                         stride=2,
                         num_residual_layers=num_residual_layers,
                         dropout_rate=dropout_rate), # H
        )
        