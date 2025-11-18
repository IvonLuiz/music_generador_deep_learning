from typing import Iterable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vector_quantizer import VectorQuantizer


class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 conv_filters: Iterable[int],
                 conv_kernels: Iterable[int],
                 conv_strides: Iterable[Tuple[int, int]],
                 latent_space_dim: int):
        super().__init__()
        layers = []
        c_in = in_channels
        for out_ch, k, s in zip(conv_filters, conv_kernels, conv_strides):
            # Normalize stride to tuple if it's an int
            s_tuple = (s, s) if isinstance(s, int) else s
            layers.append(nn.Conv2d(c_in, out_ch, kernel_size=k, stride=s_tuple, padding=k // 2))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            c_in = out_ch
        # Project to latent dim D (channels==latent_space_dim)
        layers.append(nn.Conv2d(c_in, latent_space_dim, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self,
                 out_channels: int,
                 conv_filters: Iterable[int],
                 conv_kernels: Iterable[int],
                 conv_strides: Iterable[Tuple[int, int]],
                 latent_space_dim: int):
        super().__init__()
        # Mirror of encoder: start from latent D and go through reversed filters
        filters_rev = list(conv_filters)[::-1]
        kernels_rev = list(conv_kernels)[::-1]
        strides_rev = list(conv_strides)[::-1]

        layers = []
        c_in = latent_space_dim
        for out_ch, k, s in zip(filters_rev, kernels_rev, strides_rev):
            # Normalize stride to tuple if it's an int
            s_tuple = (s, s) if isinstance(s, int) else s
            layers.append(nn.ConvTranspose2d(c_in, out_ch, kernel_size=k, stride=s_tuple, padding=k // 2, output_padding=(s_tuple[0] - 1, s_tuple[1] - 1)))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            c_in = out_ch
        # Final layer to reconstruct single-channel spectrogram (1 channel)
        layers.append(nn.ConvTranspose2d(c_in, out_channels, kernel_size=3, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x_hat = self.net(z)
        # Spectrograms are normalized 0..1
        return torch.sigmoid(x_hat)


class VQ_VAE(nn.Module):
    def __init__(self,
                 input_shape: Tuple[int, int, int],  # (H, W, C)
                 conv_filters: Iterable[int],
                 conv_kernels: Iterable[int],
                 conv_strides: Iterable[Tuple[int, int]],
                 latent_space_dim: int,
                 embeddings_size: int = 128,
                 beta: float = 0.25):
        super().__init__()
        H, W, C = input_shape
        assert C == 1, "Expected single-channel input (spectrogram)."

        self.encoder = Encoder(
            in_channels=C,
            conv_filters=conv_filters,
            conv_kernels=conv_kernels,
            conv_strides=conv_strides,
            latent_space_dim=latent_space_dim,
        )
        self.vq = VectorQuantizer(num_embeddings=embeddings_size, embedding_dim=latent_space_dim, beta=beta)
        self.decoder = Decoder(
            out_channels=C,
            conv_filters=conv_filters,
            conv_kernels=conv_kernels,
            conv_strides=conv_strides,
            latent_space_dim=latent_space_dim,
        )

    def forward(self, x):
        # x: (B, C=1, H, W) in [0,1]
        z_e = self.encoder(x)
        z_q, indices, vq_loss = self.vq(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, z_q, vq_loss

    def reconstruct(self, x):
        self.eval()
        with torch.no_grad():
            x_hat, z_q, _ = self.forward(x)
        return x_hat, z_q


def vqvae_loss(x, x_hat, vq_loss, variance: float = 1.0):
    # Likelihood term ~ scaled MSE (similar to TF code using data variance)
    recon = F.mse_loss(x_hat, x)
    return recon / (2 * variance) + vq_loss
