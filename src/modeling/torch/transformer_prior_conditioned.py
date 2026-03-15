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
        is_upsampler: bool = False,
        cond_num_embeddings: Optional[int] = None,
        upsample_stride: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """!
        @brief Initializes the TransformerPrior model.
        
        @param num_embeddings The vocabulary size (number of discrete VQ indices).
        @param model_dim The dimensionality of the token and position embeddings. If this is an upsampler prior,
        it must match the output dimension of the WaveNetConditioner.This is because the conditioning information
        from the previous level will be added to the token embeddings before being fed into the transformer layers.
        For non-upsampler priors, this can be set independently. (model_dim == cond_embedding_dim is not strictly required, but it's common to keep them the same for simplicity).
        @param num_heads The number of attention heads in the transformer encoder layers.
        @param num_layers The number of transformer encoder layers.
        @param dim_feedforward The dimensionality of the feedforward network model.
        @param max_seq_len The maximum sequence length the model can process.
        @param is_upsampler Whether this prior is an upsampler (i.e., conditions on another prior's output).
        This is used to determine whether conditioning embeddings are expected. Defaults to False.
        @param cond_num_embeddings The vocabulary size for the conditioning input (if any).
        Comes from the output of the previous level prior. Defaults to None (no conditioning).
        @param cond_embedding_dim The dimensionality of the conditioning embeddings (if any).
        Comes from the output of the previous level prior. Defaults to None (no conditioning).
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

        # Embedding layers for tokens and positions        
        self.token_embedding = nn.Embedding(num_embeddings, model_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, model_dim)

        # Initialize the transformer layers
        self._init_factored_transformer_layers()
        self.norm = nn.LayerNorm(model_dim)
        self.to_logits = nn.Linear(model_dim, num_embeddings, bias=False)

        # Initialize the conditioner if this is an upsampling prior
        self.is_upsampler = is_upsampler
        if self.is_upsampler:
            if cond_num_embeddings is None or upsample_stride is None:
                raise ValueError('cond_num_embeddings and upsample_stride must be provided for upsampler priors')
        
            # instantiate the WaveNet Conditioner for processing the conditioning input
            self.conditioner = WaveNetConditioner(
                num_embeddings=cond_num_embeddings,
                embedding_dim=model_dim, # Must match the Transformer's model_dim
                upsample_stride=upsample_stride, # This should match the upsampling factor between levels
                dropout=dropout,
            )

    def _init_factored_transformer_layers(self):
        """!
        @brief Initializes the factored transformer layers for the model. This is separated into its own method for clarity and potential future customization.
        """
        attention_patterns = ['row', 'column'] # TODO: will add 'previous_row' here later

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

    def _init_standard_transformer_layers(self):
        """!
        @brief Initializes the transformer layers for the model. This is separated into its own method for clarity and potential future customization.
        """
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.num_layers)

    def forward(self, indices: torch.Tensor, upper_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the TransformerPriorConditioned model.

        @param indices The input token indices for the current level (shape: [batch_size, seq_len]).
        @param upper_indices The input token indices from the upper level prior (shape: [batch_size, upper_seq_len]).
        This is used as conditioning information for upsampler priors. Defaults to None (no conditioning).
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

        if self.is_upsampler:
            if upper_indices is None:
                raise ValueError('upper_indices must be provided for upsampler priors')
            # Process the conditioning input through the WaveNet conditioner
            cond_emb = self.conditioner(upper_indices)  # [batch_size, model_dim, seq_len]
            #cond_emb = cond_emb.permute(0, 2, 1)  # Shape: [batch_size, seq_len, model_dim]
            # Add the conditioning embeddings to the token embeddings
            x = x + cond_emb[:, :seq_len, :]  # Ensure the conditioning embeddings are added only up to the current sequence length (in case of any length mismatch)

        # pass through transformer layers
        # FactoredAttention requires seq_len to be a multiple of block_len
        # During token-by-token generation, we pad the sequence temporarily
        # x = self.transformer(x, mask=self._causal_mask(seq_len, device))
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
        input_tokens = indices[:, :-1]
        target = indices[:, 1:]
        
        # Pass upper_indices through to forward
        logits = self.forward(input_tokens, upper_indices=upper_indices)
        return F.cross_entropy(logits.reshape(-1, self.num_embeddings), target.reshape(-1))

    @torch.no_grad()
    def generate(
        self, 
        start_tokens: torch.Tensor,
        upper_indices: Optional[torch.Tensor] = None,
        seq_len: int = 64,
        temperature: float = 1.0,  
        top_k: Optional[int] = None,
        device: Optional[torch.device] = None,   
    ) -> torch.Tensor:
        """!
        @brief Generates a sequence of token indices autoregressively given an initial input and optional conditioning.
        @param start_tokens The initial input token indices (shape: [batch_size, initial_seq_len]).
        @param upper_indices The input token indices from the upper level prior for conditioning (shape: [batch_size, upper_seq_len]). Defaults to None (no conditioning).
        @param seq_len The length of the generated sequence (including the initial input). Defaults to 64.
        @param top_k The number of top logits to keep for sampling. If None or <= 0, no filtering is applied. Defaults to None.
        @paramm temperature The sampling temperature. Must be > 0. Defaults to 1.0.
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
            tokens = start_tokens.to(device).long()
        else:
            tokens = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        
        tokens = start_tokens.to(device).long()
        if tokens.shape[0] != batch_size:
            raise ValueError("start_tokens batch size does not match batch_size")
        if tokens.shape[1] > seq_len:
            raise ValueError("start_tokens length cannot exceed requested seq_len")
        
        while tokens.shape[1] < seq_len:
            # Pass upper_indices during generation
            logits = self.forward(tokens, upper_indices=upper_indices)
            
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

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """!
        @brief Creates a causal mask for self-attention to prevent attending to future tokens.
        @param seq_len The length of the sequence to generate the mask for.
        @param device The device on which to create the mask tensor.
        @return A (seq_len, seq_len) tensor mask where positions (i, j) are -inf if j > i (future tokens) and 0 otherwise.
        """
        # Shape (T, T) with -inf above the diagonal (future positions masked).
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)


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
    
    generated_tokens = model.generate(start_tokens=lower_input_tokens,
                                      upper_indices=upper_input_tokens,
                                      seq_len=seq_len, top_k=10)
    print("Generated tokens shape:", generated_tokens.shape)  # Expected shape: (batch_size
    
    print("\nAll tests passed successfully!")
