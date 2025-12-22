from typing import Tuple, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vq_vae_residual import ResidualStack
from modeling.torch.vector_quantizer import VectorQuantizer

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_layers, stride, kernel_size=4, padding=1):
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
    def __init__(self, in_channels, out_channels, num_residual_layers, stride, kernel_size=4, padding=1, final_relu=True):
        super().__init__()
        layers = []
        stride = (stride, stride) if isinstance(stride, int) else stride  # normalize stride to tuple if it's an int

        # Residual Stack before upsampling
        layers.append(ResidualStack(in_channels=in_channels,
                                    num_hiddens=in_channels,
                                    num_residual_hiddens=in_channels // 2,
                                    num_residual_layers=num_residual_layers))

        # Upsampling conv (Transpose Conv)
        layers.append(nn.ConvTranspose2d(in_channels,
                                         out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

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
                 beta: float = 0.25):
        """
        Args:
            dim_bottom: Channels for the bottom latent space.
            dim_top: Channels for the top latent space.
            embeddings_size: Size of the codebook (K).
            num_residual_layers: Number of residual layers in each block.
            num_embeddings_top: Number of embeddings in the top vector quantizer.
            num_embeddings_bottom: Number of embeddings in the bottom vector quantizer.
            beta: Commitment loss coefficient.
        """
        
        super().__init__()
        H, W, C = input_shape

        ## ENCODER PATH
        # Bottom: image -> bottom latents (e.g., 256 -> 128 -> 64)
        self.encoder_bottom = nn.Sequential(
            EncoderBlock(in_channels=C,
                         out_channels=dim_bottom // 2,
                         num_residual_layers=num_residual_layers,
                         stride=2), # H/2
            EncoderBlock(in_channels=dim_bottom // 2,
                         out_channels=dim_bottom,
                         num_residual_layers=num_residual_layers,
                         stride=2) # H/4
        )

        # Top: bottom latents -> top latents (e.g., 64 -> 32)
        self.encoder_top = nn.Sequential(
            EncoderBlock(in_channels=dim_bottom,
                         out_channels=dim_top,
                         num_residual_layers=num_residual_layers,
                         stride=2), # H/8
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
                         num_residual_layers=num_residual_layers), # H/4
        )

        # Bottom: bottom latents + upsampled top -> reconstructed image
        self.decoder_bottom = nn.Sequential(
            DecoderBlock(in_channels=dim_bottom * 2,
                         out_channels=dim_bottom // 2,
                         stride=2,
                         num_residual_layers=num_residual_layers), # H/2
            
            # Final block: H/2 -> H. 
            # We manually construct this to avoid the final ReLU/BatchNorm in DecoderBlock,
            # which would restrict the output range before the final Sigmoid.
            ResidualStack(in_channels=dim_bottom // 2,
                          num_hiddens=dim_bottom // 2,
                          num_residual_hiddens=dim_bottom // 4,
                          num_residual_layers=num_residual_layers),
            nn.ConvTranspose2d(in_channels=dim_bottom // 2,
                               out_channels=C,
                               kernel_size=4,
                               stride=2,
                               padding=1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through the hierarchical VQ-VAE.
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            x_recon: Reconstructed tensor of shape (B, C, H, W)
            total_vq_loss: Scalar VQ loss combining top and bottom quantizers
            vq_losses_details: List with entries (vq_loss, codebook_loss, commitment_loss) for top and bottom quantizers
        """
        # Encoder
        z_bottom = self.encoder_bottom(x)  # (B, dim_bottom, H/4, W/4)
        z_top = self.encoder_top(z_bottom)  # (B, dim_top, H/8, W/8)
        
        # Quantization
        ## top
        z_top = self.pre_vq_conv_top(z_top)  # (B, dim_top, H/8, W/8)
        z_top_q, _, vq_loss_top, codebook_loss_top, commitment_loss_top = self.vq_top(z_top)
        
        ## bottom
        # Note: In VQ-VAE-2, bottom can be conditioned on top during quantization,
        # but standard implementation often quantizes the bottom features directly 
        # for reconstruction training, letting the decoder learn the merger.
        z_bottom = self.pre_vq_conv_bottom(z_bottom)  # (B, dim_bottom, H/4, W/4)
        z_bottom_q, _, vq_loss_bottom, codebook_loss_bottom, commitment_loss_bottom = self.vq_bottom(z_bottom)

        # Decoder
        z_top_upsampled = self.decoder_top(z_top_q)  # (B, dim_bottom, H/4, W/4)
        z_combined = torch.cat([z_bottom_q, z_top_upsampled], dim=1)  # (B, dim_bottom*2, H/4, W/4)
        x_recon = self.decoder_bottom(z_combined)  # (B, C, H, W)
        x_recon = torch.sigmoid(x_recon)  # Assuming input images are normalized between 0 and 1
        
        # VQ Losses return
        total_vq_loss = vq_loss_top + vq_loss_bottom
        return x_recon, total_vq_loss, [(vq_loss_top, codebook_loss_top, commitment_loss_top), (vq_loss_bottom, codebook_loss_bottom, commitment_loss_bottom)]

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input x using the trained model.
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            x_recon: Reconstructed tensor of shape (B, C, H, W)
        """
        self.eval()
        with torch.no_grad():
            x_recon, _, _ = self.forward(x)
        return x_recon

def vqvae_hierarchical_loss(x, x_recon, vq_losses_top, vq_losses_bottom, variance: float = 1.0):
    # Likelihood term ~ scaled MSE
    recon = F.mse_loss(x_recon, x) / (2 * variance)
    vq_loss_top, codebook_loss_top, commitment_loss_top = vq_losses_top
    vq_loss_bottom, codebook_loss_bottom, commitment_loss_bottom = vq_losses_bottom
    total_vq_loss = vq_loss_top + vq_loss_bottom
    return recon + total_vq_loss, recon