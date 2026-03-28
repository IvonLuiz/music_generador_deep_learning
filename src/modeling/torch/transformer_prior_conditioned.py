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
        max_time_steps: int = 500,
        is_upsampler: bool = False,
        cond_num_embeddings: Optional[int] = None,
        upsample_stride: Optional[int] = None,
        conditioner_residual_block_width: int = 1024,
        conditioner_residual_blocks: int = 16,
        conditioner_kernel_size: int = 3,
        conditioner_conv_channels: int = 1024,
        conditioner_dilation_growth_rate: int = 3,
        conditioner_dilation_cycle: int = 8,
        dropout: float = 0.1,
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
        (chunks). Defaults to 500.
        @param is_upsampler Whether this prior is an upsampler (i.e., conditions on another prior's output).
        This is used to determine whether conditioning embeddings are expected. Defaults to False.
        @param cond_num_embeddings The vocabulary size for the conditioning input (if any).
        Comes from the output of the previous level prior. Defaults to None (no conditioning).
        @param cond_embedding_dim The dimensionality of the conditioning embeddings (if any).
        Comes from the output of the previous level prior. Defaults to None (no conditioning).
        @param upsample_stride The stride for upsampling in the WaveNet conditioner. This is the ratio between the
        input and output sequences of the conditioner. Defaults to None (no upsampling).
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

        # Embedding layers for tokens and positions        
        self.token_embedding = nn.Embedding(num_embeddings, model_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, model_dim)
        self.time_embedding = nn.Embedding(max_time_steps, model_dim)

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
            ) for num_layer in range(self.num_layers)
        ])

    def forward(
        self,
        indices: torch.Tensor,
        upper_indices: Optional[torch.Tensor] = None,
        time_ids: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass for the TransformerPriorConditioned model.

        Architecture adapted from jukebox. We pass the input token indices through token and position embeddings, add
        conditioning information if this is an upsampler prior, and then pass through a series of factored transformer
        layers before projecting to logits over the vocabulary.

        @param indices The input token indices for the current level (shape: [batch_size, seq_len]).
        @param upper_indices The input token indices from the upper level prior (shape: [batch_size, upper_seq_len]).
        This is used as conditioning information for upsampler priors. Defaults to None (no conditioning).
        @param time_ids The time step IDs for each token position (shape: [batch_size, seq_len]). This is used for time embeddings.
        @return Logits over the vocabulary for the next token prediction (shape: [batch_size, seq_len, num_embeddings]).
        """
        if indices.ndim != 2:
            raise ValueError(f"indices must have shape (B, T), got {tuple(indices.shape)}")

        batch_size, seq_len = indices.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        device = indices.device
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # base embeddings (tokens + position)
        x = self.token_embedding(indices) + self.pos_embedding(pos)

        # Add time embeddings if time_ids are provided (for upsampler priors, this should help the model learn temporal structure)
        if time_ids is not None:
            time_ids = time_ids.view(batch_size)    # (batch,)
            t_emb = self.time_embedding(time_ids)   # (batch, model_dim)
            t_emb = t_emb.unsqueeze(1)              # (batch, 1, model_dim) to add to each token position
            x = x + t_emb.expand(-1, seq_len, -1)   # (batch, seq_len, model_dim) broadcast addition of time embedding to each token position

        if self.is_upsampler:
            if upper_indices is None:
                raise ValueError('upper_indices must be provided for upsampler priors')
            # Process the conditioning input through the WaveNet conditioner
            cond_emb = self.conditioner(upper_indices)  # [batch_size, upper_seq_len, model_dim]
            # Add the conditioning embeddings to the token embeddings
            x = x + cond_emb[:, :seq_len, :]  # Ensure the conditioning embeddings are added only up to the current sequence length (in case of any length mismatch)

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

    def loss(self, indices: torch.Tensor, upper_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Next-token cross-entropy on a full sequence tensor (B, T)."""
        if indices.ndim != 2:
            raise ValueError(f"indices must have shape (B, T), got {tuple(indices.shape)}")
        if indices.shape[1] < 2:
            raise ValueError("Need sequence length >= 2 for next-token training")
        
        # For next-token prediction, the input to the model is the sequence excluding the last token,
        # and the target is the sequence excluding the first token.
        input_tokens = indices[:, :-1]  # Length: T - 1
        target = indices[:, 1:]         # Length: T - 1
        
        # Pass upper_indices through to forward
        logits = self.forward(input_tokens, upper_indices=upper_indices)
        return F.cross_entropy(logits.reshape(-1, self.num_embeddings), target.reshape(-1))

    @torch.no_grad()
    def generate(
        self, 
        batch_size: int,
        start_tokens: Optional[torch.Tensor] = None,
        upper_indices: Optional[torch.Tensor] = None,
        time_id: torch.Tensor = None,
        seq_len: int = 64,
        temperature: float = 1.0,  
        top_k: Optional[int] = None,
        device: Optional[torch.device] = None,   
    ) -> torch.Tensor:
        """!
        @brief Generates a sequence of token indices autoregressively given an initial input and optional conditioning.
        @param batch_size The number of sequences to generate in parallel.
        @param start_tokens The initial input token indices (shape: [batch_size, initial_seq_len]). Defaults to None (start with a single token).
        @param upper_indices The input token indices from the upper level prior for conditioning (shape: [batch_size, upper_seq_len]). Defaults to None (no conditioning).
        @param time_id The time ID for the current generation step. Defaults to None. This is used for time embeddings and should be provided when generating from an upsampler prior to help the model learn temporal structure.
        @param seq_len The length of the generated sequence (including the initial input). Defaults to 64.
        @param top_k The number of top logits to keep for sampling. If None or <= 0, no filtering is applied. Defaults to None.
        @paramm temperature The sampling temperature. Must be > 0. Defaults to 1.0.
        @param device The device to perform generation on. If None, uses the same device as the model's parameters. Defaults to None.
        @return A tensor of generated token indices (shape: [batch_size, generated_seq_len]) where generated_seq_len <= seq_len.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if seq_len > self.max_seq_len:
            raise ValueError(f"Requested seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")
        if self.is_upsampler and upper_indices is None:
            raise ValueError("upper_indices must be provided to generate from an upsampler.")

        if device is None:
            device = next(self.parameters()).device
        
        if start_tokens is not None:
            if start_tokens.ndim != 2:
                raise ValueError(f"start_tokens must have shape (B, T), got {tuple(start_tokens.shape)}")
            if start_tokens.shape[0] != batch_size:
                raise ValueError("start_tokens batch size does not match requested batch_size")
            tokens = start_tokens.to(device).long()
        else:
            tokens = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        
        if tokens.shape[1] > seq_len:
            raise ValueError("start_tokens length cannot exceed requested seq_len")
        
        while tokens.shape[1] < seq_len:
            # Pass upper_indices during generation
            logits = self.forward(tokens, upper_indices=upper_indices, time_ids=time_id)
            
            next_logits = logits[:, -1, :] / temperature
            next_logits_filtered = self._top_k_filter(next_logits, top_k)
            probs = F.softmax(next_logits_filtered, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            tokens = torch.cat([tokens, next_token], dim=1)
        
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
        
    # Example usage
    batch_size = 4
    seq_len = 16
    num_embeddings = 2048 # VQ-VAE codebook size
    embedding_dim = 1920
    num_heads = 8
    num_layers = 6
    dim_feedforward = 2048
    max_seq_len = 64
    upsample_stride = 4  # Ratio between the upper and lower sequence lengths
    padding = 0
    kernel_size = 3
    
    print(f"Batch Size: {batch_size}")
    print(f"Top Sequence Length: {seq_len}")
    print(f"Bottom Sequence Length: {seq_len * upsample_stride}")
    print("\n" + "="*40 + "\n")

    print("Testing non-upsampler prior (no conditioning):")
    model = TransformerPriorConditioned(
        num_embeddings=num_embeddings,
        model_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_seq_len=max_seq_len,
        is_upsampler=False,
        dropout=0.1,
    )
    #print("Model architecture:\n", model)

    # dummy input tokens for the non-upsampler prior
    input_tokens = torch.randint(0, num_embeddings, (batch_size, seq_len))
    print("Input tokens shape (non-upsampler):", input_tokens.shape)  # Expected shape: (batch_size, seq_len)
    output = model(input_tokens)
    print("Output shape (non-upsampler):", output.shape)  # Expected shape: (batch_size, seq_len, num_embeddings)
    
    assert output.shape == (batch_size, seq_len, num_embeddings), f"Shape mismatch! Expected {(batch_size, seq_len, num_embeddings)}, got {output.shape}"
    print("Non-upsampler test passed successfully!")

    print("\n" + "="*40 + "\n")
    
    # Upsampler Prior (Transformer + WaveNet Conditioner)
    print("Testing upsampler prior (with conditioning):")
    
    model_upsampler = TransformerPriorConditioned(
        num_embeddings=num_embeddings,
        model_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_seq_len=max_seq_len,
        is_upsampler=True,
        cond_num_embeddings=num_embeddings,
        upsample_stride=upsample_stride,
        dropout=0.1,
    )
    #print("Upsampler Model architecture:\n", model_upsampler)
    
    # dummy input tokens for the upsampler prior
    lower_input_tokens = torch.randint(0, num_embeddings, (batch_size, seq_len))
    upper_input_tokens = torch.randint(0, num_embeddings, (batch_size, seq_len // upsample_stride))  # shorter sequence for the upper level
    print("Input tokens shape (upsampler):", lower_input_tokens.shape)  # Expected shape: (batch_size, seq_len)
    print("Upper input tokens shape (conditioning):", upper_input_tokens.shape)  # Expected shape: (batch_size, seq_len // upsample_stride)
    output = model(indices=lower_input_tokens, upper_indices=upper_input_tokens)
    print("Output shape (upsampler):", output.shape)  # Expected shape: (batch_size, seq_len, num_embeddings)
    
    assert output.shape == (batch_size, seq_len, num_embeddings), f"Shape mismatch! Expected {(batch_size, seq_len, num_embeddings)}, got {output.shape}"
    print("Upsampler test passed successfully!")
    
    print("\n" + "="*40 + "\n")
    # loss test
    print("Testing loss computation for upsampler prior:")
    loss = model.loss(lower_input_tokens, upper_indices=upper_input_tokens)
    print(f"Loss computed successfully: {loss.item()}")
    
    # test generation
    print("\nTesting generation for upsampler prior:")
    
    generated_tokens = model.generate(batch_size=batch_size, 
                                      start_tokens=lower_input_tokens,
                                      upper_indices=upper_input_tokens,
                                      seq_len=seq_len, top_k=10)
    print("Generated tokens shape:", generated_tokens.shape)  # Expected shape: (batch_size
    
    print("\nAll tests passed successfully!")
