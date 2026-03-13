import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveNetResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, dropout):
        super().__init__()
        out_channels = in_channels  # For residual connection, output channels must match input channels
    
        # Simetric padding for non causal convolution: padding = (kernel_size - 1) * dilation
        padding = (kernel_size - 1) // 2 * dilation
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Conv1d(out_channels, in_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.proj(x)
        return x + residual


class WaveNetConditioner(nn.Module):
    """!
    @brief WaveNet-based conditioner for autoregressive modeling of discrete tokens.
    
    @details This module implements a WaveNet architecture to model the conditional distribution 
    of discrete tokens (e.g., from a VQ-VAE) given some conditioning information (e.g., class labels, 
    speaker embeddings). It uses dilated causal convolutions to capture long-range dependencies in the token sequence.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_layers: int = 16,
        num_channels: int = 1024,
        kernel_size: int = 3,
        dilation_growth: int = 3,
        dilation_cycle: int = 8,
        upsample_stride: int = 4,
        dropout: float = 0.1,
    ):
        """!
        @brief Initializes the WaveNetConditioner model.
        
        @param num_embeddings The vocabulary size (number of discrete VQ indices).
        @param embedding_dim The dimensionality of the token embeddings.
        @param num_layers The number of dilated convolutional layers.
        @param num_channels The number of channels in the convolutional layers.
        @param kernel_size The kernel size for the convolutions. Defaults to 3.
        @param dilation_growth The growth factor for the dilation. Defaults to 3.
        @param dilation_cycle The cycle length for the dilation. Defaults to 8.
        @param upsample_stride The stride for the upsampling convolution.
        Ratio between the hop_length of the upsampled signal and the original signal. Defaults to 4.
        @param dropout The dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # project discrete tokens to embedding space
        self.token_embedding = nn.Embedding(num_embeddings, num_channels)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # calculate dilation for layer: 1, 3, 9, 27, ... (if growth by factor of 3)
            dilation = dilation_growth ** (i % dilation_cycle)
            self.layers.append(
                WaveNetResidualBlock(num_channels, kernel_size, dilation, dropout)
            )
        
        # Upsampling layer to match the temporal resolution of the target sequence (e.g., for audio, this would upsample from the token rate to the audio sample rate)
        # also project back down to embedding_dim
        self.upsample = nn.ConvTranspose1d(
            in_channels=num_channels, out_channels=embedding_dim,
            kernel_size=upsample_stride, stride=upsample_stride, padding=0 # no padding for transposed convolution since we want to exactly upsample by the stride factor
        )
        print(f"Upsample layer shape: {self.upsample.weight.shape}")

        # layer norm and final projection to logits before summing with the embeddings from the inferior transformer
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        """!
        @brief Forward pass of the WaveNetConditioner.
        @param x A tensor of shape (Batch, Seq_len) containing discrete token indices.
        @return A tensor of shape (Batch, Seq_len', num_channels) containing the conditioned representations,
        where Seq_len' is the upsampled sequence length.
        """
        # x is expected to be a sequence of token indices (Batch, Seq_len)
        x = self.token_embedding(x)  # (B, T, E)
        x = x.permute(0, 2, 1)  # (B, E, T) for Conv1d
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.upsample(x)  # (B, C, T')
        
        x = x.permute(0, 2, 1)  # (B, T', C) for LayerNorm matching the output of the inferior transformer
        x = self.layer_norm(x)  # (B, T', C)
        
        return x  # return the conditioned representation to be combined with the inferior transformer's output before final projection to logits

if __name__ == "__main__":
    # Example usage
    batch_size = 4
    seq_len = 16
    num_embeddings = 512
    embedding_dim = 1920
    num_channels = 1024
    upsample_stride = 4  # Ratio between the upper and lower sequence lengths
    padding = 0
    kernel_size = 3

    model = WaveNetConditioner(
        num_embeddings=num_embeddings, 
        embedding_dim=embedding_dim,
        num_channels=num_channels,
        upsample_stride=upsample_stride,
        padding=padding,
        kernel_size=kernel_size
    )
    print("Model architecture:\n", model)
    
    # dummy input tokens
    input_tokens = torch.randint(0, num_embeddings, (batch_size, seq_len))
    print("Input tokens shape:", input_tokens.shape)  # Expected shape: (batch_size, seq_len * upsample_stride, embedding_dim)
    
    # $L_{out} = (L_{in} - 1) \times \text{stride} - 2 \times \text{padding} + \text{kernel\_size}$
    # L_out = L_in * stride - 2 * padding + kernel_size
    # L_{out} = (16 - 1) * 4 - 2 * 0 + 4 = 64
    output = model(input_tokens)
    print("Output shape:", output.shape)  # Expected shape: (batch_size, seq_len * upsample_stride, embedding_dim)
    
    expected_shape = (batch_size, seq_len * upsample_stride, embedding_dim)
    assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
    
    print("Test passed successfully!")