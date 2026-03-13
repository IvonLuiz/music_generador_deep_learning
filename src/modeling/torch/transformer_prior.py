from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerPrior(nn.Module):
    """!
    @brief Causal autoregressive Transformer prior over discrete VQ indices.
    
    @details This module implements a standard decoder-only transformer architecture 
    to model the prior distribution of discrete tokens (e.g., from a VQ-VAE). 
    It predicts the next token in a sequence given the previous tokens.
    """

    def __init__(
        self,
        num_embeddings: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        """!
        @brief Initializes the TransformerPrior model.
        
        @param num_embeddings The vocabulary size (number of discrete VQ indices).
        @param model_dim The dimensionality of the token and position embeddings.
        @param num_heads The number of attention heads in the transformer encoder layers.
        @param num_layers The number of transformer encoder layers.
        @param dim_feedforward The dimensionality of the feedforward network model.
        @param max_seq_len The maximum sequence length the model can process.
        @param dropout The dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(num_embeddings, model_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        self.to_logits = nn.Linear(model_dim, num_embeddings, bias=False)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        # Shape (T, T) with -inf above the diagonal (future positions masked).
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    @staticmethod
    def _top_k_filter(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
        if top_k is None or top_k <= 0 or top_k >= logits.shape[-1]:
            return logits
        kth_vals = torch.topk(logits, k=top_k, dim=-1).values[..., -1].unsqueeze(-1)
        return torch.where(logits < kth_vals, torch.full_like(logits, float("-inf")), logits)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: Long tensor (B, T) with token ids.
        Returns:
            logits: Float tensor (B, T, V)
        """
        if indices.ndim != 2:
            raise ValueError(f"indices must have shape (B, T), got {tuple(indices.shape)}")

        batch_size, seq_len = indices.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        device = indices.device
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        x = self.token_embedding(indices) + self.pos_embedding(pos)
        mask = self._causal_mask(seq_len=seq_len, device=device)
        x = self.transformer(x, mask=mask)
        x = self.norm(x)
        return self.to_logits(x)

    def loss(self, indices: torch.Tensor) -> torch.Tensor:
        """Next-token cross-entropy on a full sequence tensor (B, T)."""
        if indices.ndim != 2:
            raise ValueError(f"indices must have shape (B, T), got {tuple(indices.shape)}")
        if indices.shape[1] < 2:
            raise ValueError("Need sequence length >= 2 for next-token training")

        inp = indices[:, :-1]
        target = indices[:, 1:]
        logits = self.forward(inp)
        return F.cross_entropy(logits.reshape(-1, self.num_embeddings), target.reshape(-1))

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        seq_len: int,
        start_tokens: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Autoregressively sample token ids.
        Returns:
            Long tensor (B, seq_len)
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if seq_len > self.max_seq_len:
            raise ValueError(f"Requested seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        if device is None:
            device = next(self.parameters()).device

        if start_tokens is not None:
            if start_tokens.ndim != 2:
                raise ValueError("start_tokens must have shape (B, T0)")
            tokens = start_tokens.to(device).long()
        else:
            tokens = torch.zeros((batch_size, 1), dtype=torch.long, device=device)

        if tokens.shape[0] != batch_size:
            raise ValueError("start_tokens batch size does not match batch_size")
        if tokens.shape[1] > seq_len:
            raise ValueError("start_tokens length cannot exceed requested seq_len")

        while tokens.shape[1] < seq_len:
            logits = self.forward(tokens)
            next_logits = logits[:, -1, :] / temperature
            next_logits = self._top_k_filter(next_logits, top_k)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens