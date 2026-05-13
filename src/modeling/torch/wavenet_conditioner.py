from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


def _is_cudnn_runtime_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return 'cudnn' in msg and ('internal_error' in msg or 'execution_failed' in msg or 'not_supported' in msg)


Stride = Union[int, Tuple[int, int]]


def _normalize_1d_stride(upsample_stride: Stride) -> int:
    if isinstance(upsample_stride, (tuple, list)):
        if len(upsample_stride) != 2:
            raise ValueError(f"Expected 2D upsample_stride tuple, got {upsample_stride}")
        stride = int(upsample_stride[0]) * int(upsample_stride[1])
    else:
        stride = int(upsample_stride)
    if stride <= 0:
        raise ValueError(f"upsample_stride must be positive, got {upsample_stride}")
    return stride


def _normalize_2d_stride(upsample_stride: Stride) -> Tuple[int, int]:
    if not isinstance(upsample_stride, (tuple, list)) or len(upsample_stride) != 2:
        raise ValueError(
            "2D conditioner requires upsample_stride=(time_stride, freq_stride), "
            f"got {upsample_stride}"
        )
    stride = (int(upsample_stride[0]), int(upsample_stride[1]))
    if stride[0] <= 0 or stride[1] <= 0:
        raise ValueError(f"upsample_stride values must be positive, got {upsample_stride}")
    return stride


class WaveNetResidualBlock(nn.Module):
    """!
    @brief A single residual block used in the WaveNetConditioner, consisting of a dilated convolution followed by a
    projection back to the residual width for the skip connection.
    """
    def __init__(
        self,
        residual_width: int,
        conv_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_2d: bool = True,
    ):
        """!
        @brief Initializes the WaveNetResidualBlock.
        
        @param residual_width The width of the residual connection.
        @param conv_channels The number of channels in the convolutional layers.
        @param kernel_size The kernel size for the convolutions.
        @param dilation The dilation factor for the dilated convolution.
        @param dropout The dropout probability.
        @param use_2d Whether to use 2D convolutions (for time-frequency conditioning) or 1D convolutions (legacy).
        """
        super().__init__()
        self.use_2d = bool(use_2d)

        # Symmetric padding for non causal convolution: padding = (kernel_size - 1) * dilation
        padding = (kernel_size - 1) // 2 * dilation

        # Dilated convolution layer
        # For 2D convolutions, we only dilate in the time dimension,
        # so we use (dilation, 1) and adjust padding accordingly
        # Also project back from 'conv_channels' to 'residual_width' for the residual connection
        if self.use_2d:
            freq_padding = (kernel_size - 1) // 2
            self.conv = nn.Conv2d(
                in_channels=residual_width,
                out_channels=conv_channels,
                kernel_size=(kernel_size, kernel_size),
                dilation=(dilation, 1),
                padding=(padding, freq_padding),
            )
            self.proj = nn.Conv2d(
                in_channels=conv_channels,
                out_channels=residual_width,
                kernel_size=1,
            )
        else:
            self.conv = nn.Conv1d(
                in_channels=residual_width,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            )
            self.proj = nn.Conv1d(
                in_channels=conv_channels,
                out_channels=residual_width,
                kernel_size=1,
            )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """!
        @brief Forward pass of the WaveNetResidualBlock.
        
        @param x A tensor of shape (B, C, T, F) for 2D mode or (B, C, S) for legacy 1D mode.
        @return A tensor of the same shape as x containing the output of the block after adding the residual connection.
        """
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
    speaker embeddings). It uses non-causal (symmetric) dilated convolutions to capture long-range
    dependencies in the upper-level token sequence, as the entire conditioning sequence is fully
    available during ancestral sampling. The output is upsampled to match the temporal resolution
    of the target sequence (e.g., audio samples) and can be combined with the output of an inferior
    transformer before final projection to logits.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        cond_freq_bins: Optional[int] = None,
        num_layers: int = 16,
        num_channels: int = 1024,
        kernel_size: int = 3,
        conv_channels: int = 1024,
        dilation_growth: int = 3,
        dilation_cycle: int = 8,
        upsample_stride: Stride = (2, 2),
        dropout: float = 0.1,
        use_2d_conditioner: bool = True,
    ):
        """!
        @brief Initializes the WaveNetConditioner model.
        
        @param num_embeddings The vocabulary size (number of discrete VQ indices).
        @param embedding_dim The dimensionality of the token embeddings.
        @param cond_freq_bins The number of frequency bins (block_len) of the conditioning grid.
        @param num_layers The number of dilated convolutional layers.
        @param num_channels The number of channels in the convolutional layers.
        @param kernel_size The kernel size for the convolutions. Defaults to 3.
        @param conv_channels The number of channels in the convolutional layers. Defaults to 1024.
        @param dilation_growth The growth factor for the dilation. Defaults to 3.
        @param dilation_cycle The cycle length for the dilation. Defaults to 8.
        @param upsample_stride The stride for the upsampling convolution.
        Ratio between the hop_length of the upsampled signal and the original signal. Defaults to 4.
        @param dropout The dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.use_2d_conditioner = bool(use_2d_conditioner)

        # project discrete tokens to embedding space
        self.token_embedding = nn.Embedding(num_embeddings, num_channels)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # calculate dilation for layer: 1, 3, 9, 27, ... (if growth by factor of 3)
            dilation = dilation_growth ** (i % dilation_cycle)
            self.layers.append(
                WaveNetResidualBlock(
                    residual_width=num_channels,
                    conv_channels=conv_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    use_2d=self.use_2d_conditioner,
                )
            )

        # Upsampling layer to match the temporal resolution of the target sequence
        # (e.g., for audio, this would upsample from the token rate to the audio sample rate)
        # also project back down to embedding_dim
        # NOTE: no padding for transposed convolution since we want to exactly upsample by the stride factor
        self.cond_freq_bins = int(cond_freq_bins) if cond_freq_bins is not None else None
        if self.use_2d_conditioner:
            if self.cond_freq_bins is None or self.cond_freq_bins <= 0:
                raise ValueError("cond_freq_bins must be provided for the 2D conditioner")
            stride_2d = _normalize_2d_stride(upsample_stride)
            self.upsample = nn.ConvTranspose2d(
                in_channels=num_channels,
                out_channels=embedding_dim,
                kernel_size=stride_2d,
                stride=stride_2d,
                padding=0,
            )
        else:
            stride_1d = _normalize_1d_stride(upsample_stride)
            self.upsample = nn.ConvTranspose1d(
                in_channels=num_channels,
                out_channels=embedding_dim,
                kernel_size=stride_1d,
                stride=stride_1d,
                padding=0,
            )

        # layer norm and final projection to logits before summing with the embeddings from the inferior transformer
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        """!
        @brief Forward pass of the WaveNetConditioner.
        @param x A tensor of shape (Batch, Seq_len) containing discrete token indices.
        @return A tensor of shape (Batch, Seq_len', embedding_dim) containing the conditioned
        representations, where Seq_len' is the upsampled sequence length and embedding_dim is
        the dimension of the embedding space (model_dim from the transformer).
        """
        # x is expected to be a sequence of token indices (Batch, Seq_len)
        B, S = x.shape
        if self.use_2d_conditioner:
            cond_freq_bins = self.cond_freq_bins
            cond_time_frames = S // cond_freq_bins
            if S != cond_time_frames * cond_freq_bins:
                raise ValueError(f"Sequence length {S} is not divisible by freq_bins {cond_freq_bins}")

            x = self.token_embedding(x)  # (B, T*F, C)
            x = x.view(B, cond_time_frames, cond_freq_bins, -1)
            x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, T, F)
        else:
            x = self.token_embedding(x)  # (B, S, C)
            x = x.permute(0, 2, 1)  # (B, C, S)

        # Retry cuDNN-sensitive dilated convolutions without cuDNN when needed.
        for layer in self.layers:
            try:
                x = layer(x)
            except RuntimeError as exc:
                print(f"RuntimeError in WaveNetResidualBlock: {exc}")
                if not x.is_cuda or not _is_cudnn_runtime_error(exc):
                    raise
                with torch.backends.cudnn.flags(enabled=False):
                    x = layer(x)
        
        try:
            x = self.upsample(x)
        except RuntimeError as exc:
            print(f"RuntimeError in WaveNetConditioner: {exc}")
            if not x.is_cuda or not _is_cudnn_runtime_error(exc):
                raise
            with torch.backends.cudnn.flags(enabled=False):
                x = self.upsample(x)
        
        # Flatten back to 1D sequence (B, E, T'*F') or (B, E, S')
        x = x.flatten(2)
        x = x.permute(0, 2, 1)  # (B, T'*F', E) for LayerNorm matching the output of the inferior transformer
        x = self.layer_norm(x)  # (B, T'*F', E)
        
        return x  # return the conditioned representation to be combined with the inferior transformer's output before final projection to logits

if __name__ == "__main__":
    # Example usage
    batch_size = 4
    cond_time_frames = 16
    cond_freq_bins = 8
    seq_len = cond_time_frames * cond_freq_bins
    num_embeddings = 512
    embedding_dim = 1920
    num_channels = 1024
    upsample_stride = (2, 2)  # Ratio between the upper and lower sequence lengths for (Time, Freq)
    kernel_size = 3

    model = WaveNetConditioner(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        cond_freq_bins=cond_freq_bins,
        num_channels=num_channels,
        upsample_stride=upsample_stride,
        kernel_size=kernel_size,
        use_2d_conditioner=True,
    )
    print("Model architecture:\n", model)
    
    # dummy input tokens
    input_tokens = torch.randint(0, num_embeddings, (batch_size, seq_len))
    print("Input tokens shape:", input_tokens.shape)  # Expected shape: (batch_size, seq_len)
    
    output = model(input_tokens)
    print("Output shape:", output.shape)  # Expected shape: (batch_size, seq_len * upsample_stride[0] * upsample_stride[1], embedding_dim)
    
    expected_shape = (batch_size, seq_len * upsample_stride[0] * upsample_stride[1], embedding_dim)
    assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
    
    print("Test passed successfully!")
