from typing import Tuple, Iterable, List
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Setup path to find siblings when running this file directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # src/modeling
grandparent_dir = os.path.dirname(parent_dir) # src

sys.path.append(grandparent_dir)

try:
    # When importing as a module from elsewhere in the project
    from modeling.torch.encoder import EncoderBlock
    from modeling.torch.decoder import DecoderBlock
    from modeling.torch.ema_vector_quantizer import EMAVectorQuantizer
except ImportError:
    # When running this script directly
    from encoder import EncoderBlock
    from decoder import DecoderBlock
    from ema_vector_quantizer import EMAVectorQuantizer


class JukeboxVQVAE(nn.Module):
    """
    Independent VQ-VAE model. To replicate Jukebox, instantiate strictly separate versions of this class
    for Bottom, Middle, and Top levels with different 'levels' (depth) parameters.
    
    This implementation on the paper Jukebox: A Generative Model for Music (https://arxiv.org/abs/).
    """

    def __init__(self,
                 input_channels: int,
                 hidden_dim: int,
                 levels: int,
                 num_residual_layers: int = 2,
                 num_embeddings: int = 2048,
                 embedding_dim: int = 64,
                 beta: float = 0.25,
                 conv_type: int = 2,
                 activation_layer: torch.Optional[nn.Module] = None,
                 dilation_growth_rate: int = 3,
                 channel_growth: int = 1,
                 ema_decay: float = 0.99,
                 epsilon: float = 1e-5,
                 restart_threshold: float = 0.5
                 ):
        """
        Args:
            input_channels: Number of input channels (e.g., 1 for mono audio, 2 for stereo, 1 for spectrogram).
            hidden_dim: Number of hidden channels in encoder/decoder.
            levels: Number of downsampling levels (Jukebox: Bottom=3, Mid=5, Top=7 approx).
            num_residual_layers: ResBlocks per level.
            conv_type: 1 for 1D (Audio), 2 for 2D (Spectrograms).
            activation_layer: Optional activation after final decoder layer (e.g., Tanh for raw audio, Sigmoid for spectrograms).
            dilation_growth_rate: Factor to grow dilation in residual stack (Jukebox uses 3).
            channel_growth: Channel multiplier per downsampling level. For Jukebox-style stability,
                            keep this at 1 (constant width).
            ema_decay: Decay factor for EMA updates in the vector quantizer.
            epsilon: Small constant for numerical stability in EMA updates.
            restart_threshold: Threshold for triggering random restarts in the codebook to prevent collapse.
        """
        if conv_type != 1 and conv_type != 2:
            raise ValueError("conv_type must be either 1 (Conv1d) or 2 (Conv2d)")
        if channel_growth < 1:
            raise ValueError("channel_growth must be >= 1")
        
        super().__init__()
        self.levels = levels
        self.conv_type = conv_type
        self.activation_layer = activation_layer
        self.channel_growth = channel_growth
        Conv = nn.Conv1d if conv_type == 1 else nn.Conv2d

        ## ENCODERS
        # Jukebox: "Each encoder block consists of a downsampling convolution, a residual network...
        # resampling convolutions use a kernel size of 4 and stride 2"
        encoder_layers = []

        # initial convolution to project input to hidden_dim channels
        encoder_layers.append(Conv(input_channels, hidden_dim, kernel_size=3, padding=1))

        current_dim = hidden_dim
        
        # Stack levels
        for level in range(levels):
            next_dim = current_dim * self.channel_growth
            encoder_layers.append(
                EncoderBlock(in_channels=current_dim,
                             out_channels=next_dim,
                             num_residual_layers=num_residual_layers,
                             stride=2,         # standard stride of 2 for downsampling
                             kernel_size=4,
                             padding=1,
                             conv_type=conv_type,
                             num_downsample_blocks=1, # done per loop iteration to allow for different conv types per level if needed
                             dilation_growth_rate=dilation_growth_rate
                             )
                )
            current_dim = next_dim
        
        self.encoder = nn.Sequential(*encoder_layers)

        ## PRE-QUANT CONVS
        # conv to adjust channels before quantization if needed
        self.pre_vq_conv = Conv(in_channels=current_dim,
                                out_channels=embedding_dim,
                                kernel_size=1)

        ## VECTOR QUANTIZERS
        self.vq = EMAVectorQuantizer(num_embeddings=num_embeddings,
                                     embedding_dim=embedding_dim,
                                     beta=beta,
                                     ema_decay=ema_decay,
                                     epsilon=epsilon,
                                     restart_threshold=restart_threshold)

        ## DECODERS
        decoder_layers = []
        decoder_layers.append(Conv(embedding_dim, current_dim, kernel_size=3, padding=1))
        
        for level in reversed(range(self.levels)):
            next_dim = current_dim // self.channel_growth if self.channel_growth > 1 else current_dim
            decoder_layers.append(
                DecoderBlock(in_channels=current_dim,
                             out_channels=next_dim,
                             num_residual_layers=num_residual_layers,
                             stride=2,         # standard stride of 2 for upsampling
                             kernel_size=4,
                             padding=1,
                             conv_type=conv_type,
                             num_downsample_blocks=1, # done per loop iteration to allow for different conv types per level if needed
                             dilation_growth_rate=dilation_growth_rate
                             )
                )
            current_dim = next_dim

        # Final block for projection back to input channels after all upsampling is done
        decoder_layers.append(Conv(current_dim, input_channels, kernel_size=3, padding=1))

        # Final activation? Jukebox raw audio usually entails no activation (Linear) or Tanh.
        # For spectrograms, we want a sigmoid
        self.decoder = nn.Sequential(*decoder_layers)

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
        z = self.encoder(x)  # (B, hidden_dim, H/2^levels, W/2^levels)
        
        # Quantization
        z = self.pre_vq_conv(z)
        z_q, _, vq_loss, codebook_loss, commitment_loss = self.vq(z)

        # Decoder
        x_recon = self.decoder(z_q)

        # VQ Losses return
        total_vq_loss = vq_loss

        # Apply activation layer if it exists
        if self.activation_layer is not None:
            x_recon = self.activation_layer(x_recon)

        return x_recon, total_vq_loss, [(vq_loss, codebook_loss, commitment_loss)]


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


if __name__ == "__main__":
    # Test block
    print("Initializing Jukebox VQ-VAE...")
    
    # 1D Audio Test
    model_1d = JukeboxVQVAE(input_channels=1, hidden_dim=64, levels=3, conv_type=1)
    x_1d = torch.randn(2, 1, 8000) # (Batch, Channels, Time)
    y_1d, _, _ = model_1d(x_1d)
    print(f"1D Input: {x_1d.shape}, Output: {y_1d.shape}")
    
    # 2D Spectrogram Test
    model_2d = JukeboxVQVAE(input_channels=1, hidden_dim=64, levels=2, conv_type=2, 
                           activation_layer=nn.Sigmoid())
    x_2d = torch.randn(2, 1, 128, 128) # (Batch, Channels, H, W)
    y_2d, _, _ = model_2d(x_2d)
    print(f"2D Input: {x_2d.shape}, Output: {y_2d.shape}")