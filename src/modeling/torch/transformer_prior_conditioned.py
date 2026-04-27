import os
import sys
from typing import Optional

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
    from modeling.torch.wavenet_conditioner import WaveNetConditioner
    from modeling.torch.factored_transformer_layer import FactoredTransformerLayer
except ImportError:
    # When running this script directly
    from wavenet_conditioner import WaveNetConditioner
    from factored_transformer_layer import FactoredTransformerLayer


class TransformerPriorConditioned(nn.Module):
    """
    Transformer prior conditioned on other transformer priors outputs.

    "train separate models for the top-level prior p(z top), and upsamplers
    p(z middle|z top) and p(z bottom|z middle, z top). Each of these is an
    autoregressive modeling problem in the discrete token space produced by
    the VQ-VAE. We use Transformers with sparse attention (Vaswani et al., 2017; Child
    et al., 2019) as they are currently the SOTA in autoregressive modeling.
    We propose a simplified version which we call the Scalable Transformer,
    that is easier to implement and scale .
    """

    def __init__(
        self,
        num_embeddings: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        max_seq_len: int,
        block_len: int = 16,
        max_time_steps: int = 100,
        is_upsampler: bool = False,
        cond_num_embeddings: Optional[int] = None,
        upsample_stride: Optional[int] = None,
        second_cond_num_embeddings: Optional[int] = None,
        second_upsample_stride: Optional[int] = None,
        use_bos_token: bool = False,
        conditioner_residual_block_width: int = 1024,
        conditioner_residual_blocks: int = 16,
        conditioner_kernel_size: int = 3,
        conditioner_conv_channels: int = 1024,
        conditioner_dilation_growth_rate: int = 3,
        conditioner_dilation_cycle: int = 8,
        dropout: float = 0.1,
        attention_qkv_ratio: float = 1.0,
    ):
        """!
        @brief Initializes the TransformerPrior model.
        
        @param num_embeddings The vocabulary size (number of discrete VQ indices).
        @param model_dim The dimensionality of the token and position embeddings. If this is an upsampler prior,
        it must match the output dimension of the WaveNetConditioner.This is because the conditioning information
        from the previous level will be added to the token embeddings before being fed into the transformer layers.
        For non-upsampler priors, this can be set independently. (model_dim == cond_embedding_dim is not strictly
        required, but it's common to keep them the same for simplicity).
        @param num_heads The number of attention heads in the transformer encoder layers.
        @param num_layers The number of transformer encoder layers.
        @param dim_feedforward The dimensionality of the feedforward network model.
        @param max_seq_len The maximum sequence length the model can process.
        @param max_time_steps The maximum number of time steps the model can process. This is used for the time
        embedding and should be set based on the expected maximum length of the input sequences in terms of time steps
        (chunks). Defaults to 100.
        @param is_upsampler Whether this prior is an upsampler (i.e., conditions on another prior's output).
        This is used to determine whether conditioning embeddings are expected. Defaults to False.
        @param cond_num_embeddings The vocabulary size for the conditioning input (if any).
        Comes from the output of the previous level prior. Defaults to None (no conditioning).
        @param cond_embedding_dim The dimensionality of the conditioning embeddings (if any).
        Comes from the output of the previous level prior. Defaults to None (no conditioning).
        @param upsample_stride The stride for upsampling in the WaveNet conditioner. This is the ratio between the
        input and output sequences of the conditioner. Defaults to None (no upsampling).
        @param use_bos_token Whether to reserve an extra input token ID for beginning-of-sequence conditioning.
        When enabled, token embeddings accept IDs in [0, num_embeddings] and generation starts from bos_token_id.
        @param conditioner_residual_block_width The number of channels in the residual blocks of the WaveNet conditioner. Defaults to 1024.
        @param conditioner_residual_blocks The number of residual blocks in the WaveNet conditioner. Defaults to 16.
        @param conditioner_kernel_size The kernel size for the WaveNet conditioner. Defaults to 3.
        @param conditioner_conv_channels The number of channels in the convolutional layers of the WaveNet conditioner. Defaults to 1024.
        @param conditioner_dilation_growth_rate The rate at which the dilation factor grows in the WaveNet conditioner. Defaults to 3.
        @param conditioner_dilation_cycle The cycle length for the dilation factors in the WaveNet conditioner. Defaults to 8.
        @param dropout The dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.max_seq_len = max_seq_len
        self.block_len = block_len
        self.dropout = dropout
        self.max_time_steps = max_time_steps
        self.use_bos_token = use_bos_token
        self.attention_qkv_ratio = float(attention_qkv_ratio)
        self.bos_token_id = num_embeddings if use_bos_token else None
        self.input_vocab_size = num_embeddings + (1 if use_bos_token else 0)

        # Initialize the conditioner if this is an upsampling prior
        self.is_upsampler = is_upsampler
        if self.is_upsampler:
            if cond_num_embeddings is None or upsample_stride is None:
                raise ValueError('cond_num_embeddings and upsample_stride must be provided for upsampler priors')
        
            # instantiate the WaveNet Conditioner for processing the conditioning input
            self.conditioner = WaveNetConditioner(
                num_embeddings=cond_num_embeddings,
                embedding_dim=model_dim,
                num_layers=conditioner_residual_blocks,
                num_channels=conditioner_residual_block_width,
                kernel_size=conditioner_kernel_size,
                conv_channels=conditioner_conv_channels,
                dilation_growth=conditioner_dilation_growth_rate,
                dilation_cycle=conditioner_dilation_cycle,
                upsample_stride=upsample_stride,
                dropout=dropout,
            )

            self.second_conditioner = None
            if second_cond_num_embeddings is not None and second_upsample_stride is not None:
                self.second_conditioner = WaveNetConditioner(
                    num_embeddings=second_cond_num_embeddings,
                    embedding_dim=model_dim,
                    num_layers=conditioner_residual_blocks,
                    num_channels=conditioner_residual_block_width,
                    kernel_size=conditioner_kernel_size,
                    conv_channels=conditioner_conv_channels,
                    dilation_growth=conditioner_dilation_growth_rate,
                    dilation_cycle=conditioner_dilation_cycle,
                    upsample_stride=second_upsample_stride,
                    dropout=dropout,
                )
        else:
            self.second_conditioner = None

        # Embedding layers for tokens and positions
        self.token_embedding = nn.Embedding(self.input_vocab_size, model_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, model_dim)
        # Continuous timing projection: maps [start_time_s, total_duration_s, fraction_elapsed]
        # to model_dim and adds a global conditioning signal broadcast over the sequence.
        self.timing_proj = nn.Linear(3, model_dim)  # jukebox Section 5.2

        # Initialize the transformer layers
        self._init_factored_transformer_layers()
        self.norm = nn.LayerNorm(model_dim)
        self.to_logits = nn.Linear(model_dim, num_embeddings, bias=False)

    def _init_factored_transformer_layers(self):
        """!
        @brief Initializes the factored transformer layers for the model. This is separated into its own method for clarity and potential future customization.
        """
        attention_patterns = ['row', 'column', 'previous_row']

        self.transformer = nn.Sequential(*[
            FactoredTransformerLayer(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                block_len=self.block_len,
                attention_type = attention_patterns[num_layer % len(attention_patterns)],
                mlp_dim=self.dim_feedforward,
                dropout=self.dropout,
                qkv_ratio=self.attention_qkv_ratio,
            ) for num_layer in range(self.num_layers)
        ])

    def forward(
        self,
        indices: torch.Tensor,
        upper_indices: Optional[torch.Tensor] = None,
        second_upper_indices: Optional[torch.Tensor] = None,
        timing: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the TransformerPriorConditioned model.

        Architecture adapted from jukebox. We pass the input token indices through token and position embeddings, add
        conditioning information if this is an upsampler prior, and then pass through a series of factored transformer
        layers before projecting to logits over the vocabulary.

        @param indices The input token indices for the current level (shape: [batch_size, seq_len]).
        @param upper_indices The input token indices from the upper level prior (shape: [batch_size, upper_seq_len]).
        This is used as conditioning information for upsampler priors. Defaults to None (no conditioning).
        @param timing Optional global timing tensor of shape (batch_size, 3) with float values
        [start_time_seconds, total_duration_seconds, fraction_elapsed]. Projected to model_dim
        and broadcast over the full sequence so the model knows where it is inside a song.
        @return Logits over the vocabulary for the next token prediction (shape: [batch_size, seq_len, num_embeddings]).
        """
        if indices.ndim != 2:
            raise ValueError(f"indices must have shape (B, T), got {tuple(indices.shape)}")

        batch_size, seq_len = indices.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")
        if torch.any(indices < 0) or torch.any(indices >= self.input_vocab_size):
            raise ValueError(
                f"indices must be in [0, {self.input_vocab_size - 1}], got min={int(indices.min())}, max={int(indices.max())}"
            )

        device = indices.device
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # base embeddings (tokens + position)
        x = self.token_embedding(indices) + self.pos_embedding(pos)

        # Add global timing embedding if provided.
        # timing shape: (B, 3) with [start_time_s, total_duration_s, fraction_elapsed].
        # Projected to (B, model_dim) then broadcast over seq_len.
        if timing is not None:
            timing = timing.to(device=device, dtype=torch.float32)
            if timing.ndim == 1:
                timing = timing.unsqueeze(0)
            if timing.shape[0] == 1 and batch_size > 1:
                timing = timing.expand(batch_size, -1)
            if timing.shape != (batch_size, 3):
                raise ValueError(
                    f"timing must have shape (B, 3) or (1, 3), got {tuple(timing.shape)}"
                )
            t_emb = self.timing_proj(timing)    # (B, model_dim)
            x = x + t_emb.unsqueeze(1)          # broadcast: (B, 1, model_dim) + (B, T, model_dim)

        if self.is_upsampler:
            if upper_indices is None:
                raise ValueError('upper_indices must be provided for upsampler priors')
            upper_indices = upper_indices.to(device=device, dtype=torch.long)
            # Conditioning embedding vocabulary comes from the upper level codebook size
            cond_vocab = self.conditioner.token_embedding.weight.shape[0]

            if torch.any(upper_indices < 0):
                raise ValueError("upper_indices contains negative values")

            # Allow exactly one out-of-range ID for BOS from an upper prior with BOS enabled
            # BOS id is expected to be exactly equal to cond_vocab and is mapped to a neutral token
            if torch.any(upper_indices >= cond_vocab):
                if torch.any(upper_indices > cond_vocab):
                    raise ValueError(
                        f"upper_indices contains values above conditioning vocab (max allowed={cond_vocab})"
                    )
                upper_indices = upper_indices.clone()
                upper_indices[upper_indices == cond_vocab] = 0

            # Process the conditioning input through the WaveNet conditioner
            cond_emb = self.conditioner(upper_indices)  # [batch_size, upper_seq_len, model_dim]
            # Add the conditioning embeddings to the token embeddings
            x = x + cond_emb[:, :seq_len, :]  # Ensure the conditioning embeddings are added only up to the current sequence length (in case of any length mismatch)

            if self.second_conditioner is not None:
                if second_upper_indices is None:
                    raise ValueError('second_upper_indices must be provided when second_conditioner is enabled')

                second_upper_indices = second_upper_indices.to(device=device, dtype=torch.long)
                cond_vocab_2 = self.second_conditioner.token_embedding.weight.shape[0]

                if torch.any(second_upper_indices < 0):
                    raise ValueError("second_upper_indices contains negative values")

                if torch.any(second_upper_indices >= cond_vocab_2):
                    if torch.any(second_upper_indices > cond_vocab_2):
                        raise ValueError(
                            f"second_upper_indices contains values above conditioning vocab (max allowed={cond_vocab_2})"
                        )
                    second_upper_indices = second_upper_indices.clone()
                    second_upper_indices[second_upper_indices == cond_vocab_2] = 0

                cond_emb_2 = self.second_conditioner(second_upper_indices)
                x = x + cond_emb_2[:, :seq_len, :]

        # pass through transformer layers
        # FactoredAttention requires seq_len to be a multiple of block_len
        # During token-by-token generation, we pad the sequence temporarily
        remainder = seq_len % self.block_len
        pad_len = 0
        if remainder != 0:
            pad_len = self.block_len - remainder
            # pad format for 3D tensor (batch, seq, dim): (dim_left, dim_right, seq_left, seq_right)
            x = F.pad(x, (0, 0, 0, pad_len), value=0)  # Pad the sequence dimension with zeros
        
        # Pass through our alternating Factored Transformer layers
        x = self.transformer(x)
        
        # Slice off the padding if we added any
        if pad_len > 0:
            x = x[:, :-pad_len, :]

        x = self.norm(x)
        logits = self.to_logits(x)
        
        return logits

    def loss(
        self,
        indices: torch.Tensor,
        upper_indices: Optional[torch.Tensor] = None,
        second_upper_indices: Optional[torch.Tensor] = None,
        timing: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """!
        Next-token cross-entropy on a full sequence tensor (B, T).

        @param indices The input token indices for the current level (shape: [batch_size, seq_len]).
        @param upper_indices The input token indices from the upper level prior (shape: [batch_size, upper_seq_len]).
        This is used as conditioning information for upsampler priors. Defaults to None (no conditioning).
        @param timing Optional global timing tensor of shape (batch_size, 3) with float values
        [start_time_seconds, total_duration_seconds, fraction_elapsed]. Defaults to None.
        @return The computed cross-entropy loss for next-token prediction.
        """
        if indices.ndim != 2:
            raise ValueError(f"indices must have shape (B, T), got {tuple(indices.shape)}")
        min_seq_len = 1 if self.use_bos_token else 2
        if indices.shape[1] < min_seq_len:
            raise ValueError(f"Need sequence length >= {min_seq_len} for next-token training")
        
        # For next-token prediction, prepend BOS when enabled so the model learns to start from an empty prefix.
        if self.use_bos_token:
            bos = torch.full((indices.shape[0], 1), self.bos_token_id, dtype=indices.dtype, device=indices.device)
            input_tokens = torch.cat([bos, indices[:, :-1]], dim=1)
            target = indices
        else:
            # For next-token prediction, the input to the model is the sequence excluding the last token,
            # and the target is the sequence excluding the first token.
            input_tokens = indices[:, :-1]  # Length: T - 1
            target = indices[:, 1:]         # Length: T - 1
        
        logits = self.forward(
            input_tokens,
            upper_indices=upper_indices,
            second_upper_indices=second_upper_indices,
            timing=timing,
        )
        return F.cross_entropy(logits.reshape(-1, self.num_embeddings), target.reshape(-1))

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        start_tokens: Optional[torch.Tensor] = None,
        upper_indices: Optional[torch.Tensor] = None,
        second_upper_indices: Optional[torch.Tensor] = None,
        timing: Optional[torch.Tensor] = None,
        seq_len: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """!
        @brief Generates a sequence of token indices autoregressively given an initial input and optional conditioning.
        @param batch_size The number of sequences to generate in parallel.
        @param start_tokens The initial input token indices (shape: [batch_size, initial_seq_len]). Defaults to None.
        @param upper_indices Conditioning token indices from the upper level prior (shape: [batch_size, upper_seq_len]).
        @param second_upper_indices Second conditioning indices (bottom prior only). Defaults to None.
        @param timing Global timing tensor of shape (batch_size, 3) or (1, 3) with float values
        [start_time_seconds, total_duration_seconds, fraction_elapsed]. Defaults to None (no timing signal).
        @param seq_len The length of the generated sequence. Defaults to 64.
        @param top_k Top-k filtering. If None or <= 0, no filtering is applied. Defaults to None.
        @param temperature Sampling temperature. Must be > 0. Defaults to 1.0.
        @param device Device to perform generation on. Defaults to model's device.
        @return Generated token indices (shape: [batch_size, seq_len]).
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if seq_len > self.max_seq_len:
            raise ValueError(f"Requested seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")
        if self.is_upsampler and upper_indices is None:
            raise ValueError("upper_indices must be provided to generate from an upsampler.")
        if self.second_conditioner is not None and second_upper_indices is None:
            raise ValueError("second_upper_indices must be provided when second_conditioner is enabled.")

        if device is None:
            device = next(self.parameters()).device
        
        bos_prefix_len = 0

        if start_tokens is not None:
            if start_tokens.ndim != 2:
                raise ValueError(f"start_tokens must have shape (B, T), got {tuple(start_tokens.shape)}")
            if start_tokens.shape[0] != batch_size:
                raise ValueError("start_tokens batch size does not match requested batch_size")
            tokens = start_tokens.to(device).long()
        else:
            if self.use_bos_token:
                tokens = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
                bos_prefix_len = 1
            else:
                tokens = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        
        if tokens.shape[1] > seq_len:
            raise ValueError("start_tokens length cannot exceed requested seq_len")

        target_len = seq_len + bos_prefix_len

        while tokens.shape[1] < target_len:
            logits = self.forward(
                tokens,
                upper_indices=upper_indices,
                second_upper_indices=second_upper_indices,
                timing=timing,
            )
            
            next_logits = logits[:, -1, :] / temperature
            next_logits_filtered = self._top_k_filter(next_logits, top_k)
            probs = F.softmax(next_logits_filtered, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            tokens = torch.cat([tokens, next_token], dim=1)

        if bos_prefix_len > 0:
            tokens = tokens[:, bos_prefix_len:]

        return tokens

    @staticmethod
    def _top_k_filter(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
        """!
        @brief Filters logits to keep only the top-k values.
        @param logits The input logits tensor of shape (B, V) where V is the vocabulary size.
        @param top_k The number of top logits to keep. If None or <= 0, no filtering is applied.
        @return A tensor of the same shape as logits where values not in the top-k are set to -inf.
        """
        if top_k is None or top_k <= 0 or top_k >= logits.shape[-1]:
            return logits
        kth_vals = torch.topk(logits, k=top_k, dim=-1).values[..., -1].unsqueeze(-1)
        return torch.where(logits < kth_vals, torch.full_like(logits, float("-inf")), logits)


if __name__ == "__main__":
    torch.manual_seed(42)

    # Example usage — reflects the new level-specific window architecture.
    # All three levels produce 512 tokens; conditioners use sliced upper sequences.
    #   Top:    8×64 = 512 tokens  (full 2048-frame window)
    #   Middle: 16×32 = 512 tokens (512-frame window), conditioned on 8×16=128 sliced Top tokens
    #   Bottom: 32×16 = 512 tokens (128-frame window),  conditioned on 16×8=128 sliced Middle tokens
    # Bottom Prior (512) is conditioned on:
    #   Middle slice (128 tokens) -> Stride 4
    #   Top slice (32 tokens)     -> Stride 16
    batch_size = 4
    seq_len_bot = seq_len = 512
    seq_len_mid_slice = 128
    seq_len_top_slice = 32
    cond_seq_len = 128     # sliced upper-level conditioning (stride-4 conditioner)
    num_embeddings = 2048  # VQ-VAE codebook size
    embedding_dim = 512
    num_heads = 8
    num_layers = 6
    dim_feedforward = 2048
    max_seq_len = 512
    upsample_stride = 4    # 128 upper tokens → 512 lower tokens
    padding = 0
    kernel_size = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Batch Size: {batch_size}")
    print(f"Target Sequence Length (bottom): {seq_len_bot}")
    print(f"Target Sequence Length (middle slice): {seq_len_mid_slice}")
    print(f"Target Sequence Length (top slice): {seq_len_top_slice}")
    print(f"Conditioning Sequence Length (sliced): {cond_seq_len}")
    print("\n" + "="*40 + "\n")


    print("Testing timing Sensitivity (Top Prior)")
    model_top = TransformerPriorConditioned(
        num_embeddings=num_embeddings,
        model_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_seq_len=seq_len_bot,
    ).to(device).eval()
    
    dummy_tokens = torch.randint(0, num_embeddings, (1, 512)).to(device)
    
    # Define two different timing scenarios for the same tokens
    timing_start = torch.tensor([[0.0, 180.0, 0.0]]).to(device)   # Start of a 3min song
    timing_end = torch.tensor([[179.0, 180.0, 0.99]]).to(device) # End of a 3min song
    with torch.no_grad():
        logits_start = model_top(dummy_tokens, timing=timing_start)
        logits_end = model_top(dummy_tokens, timing=timing_end)

    # Calculate the difference in predictions
    diff = (logits_start - logits_end).abs().mean().item()
    print(f"Logit Mean Absolute Difference (Start vs End): {diff:.6f}")
    
    if diff > 1e-7:
        print("SUCCESS: Model is responsive to timing signal.")
    else:
        print("FAILURE: Model is ignoring the timing signal.")

    print("\n" + "="*40 + "\n")

    print("Testing non-upsampler prior (Top — no conditioning):")
    model = TransformerPriorConditioned(
        num_embeddings=num_embeddings,
        model_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_seq_len=max_seq_len,
        block_len=8,   # Top grid: 8 freq × 64 time
        is_upsampler=False,
        dropout=0.1,
    ).to(device)

    input_tokens = torch.randint(0, num_embeddings, (batch_size, seq_len)).to(device)
    print("Input tokens shape (top):", input_tokens.shape)
    timing = torch.tensor([[0.0, 23.8, 0.0]] * batch_size)
    output = model(input_tokens, timing=timing)
    print("Output shape (top):", output.shape)
    assert output.shape == (batch_size, seq_len, num_embeddings), f"Shape mismatch! Expected {(batch_size, seq_len, num_embeddings)}, got {output.shape}"
    print("Top prior test passed successfully!")

    print("\n" + "="*40 + "\n")

    print("Testing upsampler prior (Middle — conditioned on 128 sliced Top tokens):")
    model_upsampler = TransformerPriorConditioned(
        num_embeddings=num_embeddings,
        model_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_seq_len=max_seq_len,
        block_len=16,  # Middle grid: 16 freq × 32 time
        is_upsampler=True,
        cond_num_embeddings=num_embeddings,
        upsample_stride=upsample_stride,  # 128 → 512, stride=4
        dropout=0.1,
    ).to(device)

    lower_input_tokens = torch.randint(0, num_embeddings, (batch_size, seq_len)).to(device)
    upper_input_tokens = torch.randint(0, num_embeddings, (batch_size, cond_seq_len)).to(device)  # 128 sliced Top tokens
    print("Input tokens shape (middle):", lower_input_tokens.shape)
    print("Conditioning tokens shape (sliced Top):", upper_input_tokens.shape)
    output = model_upsampler(indices=lower_input_tokens, upper_indices=upper_input_tokens, timing=timing)
    print("Output shape (middle upsampler):", output.shape)
    assert output.shape == (batch_size, seq_len, num_embeddings), f"Shape mismatch! Expected {(batch_size, seq_len, num_embeddings)}, got {output.shape}"
    print("Middle upsampler test passed successfully!")

    # loss test
    print("\n" + "="*40 + "\n")
    print("Testing loss and generation for middle upsampler prior:")
    loss = model_upsampler.loss(lower_input_tokens, upper_indices=upper_input_tokens, timing=timing)
    print(f"Loss computed successfully: {loss.item():.4f}")

    print("\n" + "="*40 + "\n")
    print("Testing generation for middle upsampler prior:")
    generated_tokens = model_upsampler.generate(
        batch_size=batch_size,
        start_tokens=None,
        upper_indices=upper_input_tokens,
        seq_len=seq_len,
        top_k=50,
        timing=timing,
    )
    print("Generated tokens shape:", generated_tokens.shape)
    assert generated_tokens.shape == (batch_size, seq_len), f"Generated shape mismatch: {generated_tokens.shape}"

    print("\n" + "="*40 + "\n")
    print("Testing Bottom Prior Simulation (Conditioned on Middle + Top)")
    
    # Bottom upsampler parameters
    model_bot = TransformerPriorConditioned(
        num_embeddings=num_embeddings,
        model_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_seq_len=seq_len_bot,
        is_upsampler=True,
        cond_num_embeddings=num_embeddings, # Middle level codebook
        upsample_stride=4,                  # 128 -> 512
        second_cond_num_embeddings=num_embeddings, # Top level codebook
        second_upsample_stride=16,          # 32 -> 512
    ).to(device)

    # Dummy data
    bot_indices = torch.randint(0, num_embeddings, (batch_size, seq_len_bot)).to(device)
    mid_cond = torch.randint(0, num_embeddings, (batch_size, seq_len_mid_slice)).to(device)
    top_cond = torch.randint(0, num_embeddings, (batch_size, seq_len_top_slice)).to(device)
    timing = torch.tensor([[10.0, 100.0, 0.1]] * batch_size).to(device)

    # Test Forward
    logits = model_bot(
        indices=bot_indices, 
        upper_indices=mid_cond, 
        second_upper_indices=top_cond, 
        timing=timing
    )
    print(f"Bottom Prior Forward shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len_bot, num_embeddings)

    # Test Loss
    loss_val = model_bot.loss(
        indices=bot_indices, 
        upper_indices=mid_cond, 
        second_upper_indices=top_cond, 
        timing=timing
    )
    print(f"Bottom Prior Loss: {loss_val.item():.4f}")
    assert not torch.isnan(loss_val)

    # Test Generation
    print("Testing Bottom Prior Generation...")
    gen_tokens = model_bot.generate(
        batch_size=batch_size,
        upper_indices=mid_cond,
        second_upper_indices=top_cond,
        timing=timing,
        seq_len=16, # Short gen for speed
        temperature=0.8,
        top_k=50
    )
    print(f"Generated tokens shape: {gen_tokens.shape}")
    assert gen_tokens.shape == (batch_size, 16)

    print("\n" + "="*40 + "\n")
    print("\nAll tests passed successfully!")
