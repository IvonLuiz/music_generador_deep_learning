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
        use_timing_conditioning: bool = True,
        timing_num_bins: int = 1024,
        duration_num_bins: int = 256,
        timing_window_seconds: Optional[float] = None,
        timing_max_duration_seconds: float = 3600.0,
        timing_embedding_init_std: float = 0.02,
        timing_embedding_scale: float = 1.0,
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
        @param use_timing_conditioning Whether to add learned timing metadata to every token embedding.
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
        self.use_timing_conditioning = bool(use_timing_conditioning)
        self.timing_num_bins = int(timing_num_bins)
        self.duration_num_bins = int(duration_num_bins)
        self.timing_window_seconds = float(timing_window_seconds) if timing_window_seconds is not None else None
        self.timing_max_duration_seconds = float(timing_max_duration_seconds)
        self.timing_embedding_scale = float(timing_embedding_scale)
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
        self.absolute_timing_embedding = None
        self.relative_timing_embedding = None
        self.duration_timing_embedding = None
        if self.use_timing_conditioning:
            if self.timing_num_bins < 2:
                raise ValueError(f"timing_num_bins must be >= 2, got {self.timing_num_bins}")
            if self.duration_num_bins < 2:
                raise ValueError(f"duration_num_bins must be >= 2, got {self.duration_num_bins}")
            self.absolute_timing_embedding = nn.Embedding(self.timing_num_bins, model_dim)
            self.relative_timing_embedding = nn.Embedding(self.timing_num_bins, model_dim)
            self.duration_timing_embedding = nn.Embedding(self.duration_num_bins, model_dim)
            self._init_learned_timing_embeddings(float(timing_embedding_init_std))

        # Initialize the transformer layers
        self._init_factored_transformer_layers()
        self.norm = nn.LayerNorm(model_dim)
        self.to_logits = nn.Linear(model_dim, num_embeddings, bias=False)

    def _init_learned_timing_embeddings(self, init_std: float):
        """!
        @brief Initialize timing embeddings gently so they start as a cue, not a dominant signal.
        @param init_std Standard deviation used for normal initialization.
        """
        for emb in (
            self.absolute_timing_embedding,
            self.relative_timing_embedding,
            self.duration_timing_embedding,
        ):
            nn.init.normal_(emb.weight, mean=0.0, std=init_std)

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
        conditioning_emb: Optional[torch.Tensor] = None,
        second_conditioning_emb: Optional[torch.Tensor] = None,
        timing_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """!
        @brief Forward pass for the TransformerPriorConditioned model.

        @details Architecture adapted from jukebox. We pass the input token indices through token and position embeddings, add
        conditioning information if this is an upsampler prior, and then pass through a series of factored transformer
        layers before projecting to logits over the vocabulary.

        @param indices Input token indices for the current level (shape: [batch_size, seq_len]).
        @param upper_indices Upper-level token indices for upsampler priors (shape: [batch_size, upper_seq_len]).
        @param second_upper_indices Second upper-level token indices for dual-conditioning priors.
        @param timing Optional timing tensor (shape: [batch_size, 3]) with
        [start_time_seconds, total_duration_seconds, fraction_elapsed].
        @param conditioning_emb Optional precomputed conditioning embedding (shape: [batch_size, T, D]).
        @param second_conditioning_emb Optional second conditioning embedding (shape: [batch_size, T, D]).
        @param timing_emb Optional precomputed timing embedding (shape: [batch_size, T, D]).
        @return Logits over the vocabulary (shape: [batch_size, seq_len, num_embeddings]).
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

        # Add learned timing embeddings if provided.
        # timing shape: (B, 3) with [start_time_s, total_duration_s, fraction_elapsed].
        if self.use_timing_conditioning and timing_emb is not None:
            t_emb = self._align_cached_embedding(timing_emb, batch_size, seq_len, device, 'timing_emb')
            x = x + self.timing_embedding_scale * t_emb
        elif self.use_timing_conditioning and timing is not None:
            timing = timing.to(device=device, dtype=torch.float32)
            if timing.ndim == 1:
                timing = timing.unsqueeze(0)
            if timing.shape[0] == 1 and batch_size > 1:
                timing = timing.expand(batch_size, -1)
            if timing.shape != (batch_size, 3):
                raise ValueError(
                    f"timing must have shape (B, 3) or (1, 3), got {tuple(timing.shape)}"
                )
            t_emb = self._learned_timing_embedding(timing, seq_len, device)
            x = x + self.timing_embedding_scale * t_emb

        if self.is_upsampler:
            if conditioning_emb is None:
                cond_emb = self._compute_conditioning_embedding(
                    upper_indices,
                    self.conditioner,
                    device,
                    'upper_indices',
                )
            else:
                cond_emb = conditioning_emb
            cond_emb = self._align_cached_embedding(cond_emb, batch_size, seq_len, device, 'conditioning_emb')
            x = x + cond_emb    # add the conditioning embeddings to the token embeddings before feeding into the transformer layers.

            if self.second_conditioner is not None:
                if second_conditioning_emb is None:
                    cond_emb_2 = self._compute_conditioning_embedding(
                        second_upper_indices,
                        self.second_conditioner,
                        device,
                        'second_upper_indices',
                    )
                else:
                    cond_emb_2 = second_conditioning_emb
                cond_emb_2 = self._align_cached_embedding(cond_emb_2, batch_size, seq_len, device, 'second_conditioning_emb')
                x = x + cond_emb_2  # add the second conditioning embeddings if present (e.g., for the bottom prior conditioned on both middle and top)

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

    def _compute_conditioning_embedding(
        self,
        indices: Optional[torch.Tensor],
        conditioner: nn.Module,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        """!
        @brief Compute a WaveNet conditioner embedding from upper-level indices.
        @param indices Conditioning token indices from the upper level.
        @param conditioner WaveNet conditioner module to apply.
        @param device Target device for conditioning tensors.
        @param name Label used in validation errors.
        @return Conditioning embeddings shaped `(B, T, D)`.
        @throws ValueError If indices are missing or out of range.
        """
        if indices is None:
            raise ValueError(f'{name} must be provided for upsampler priors')

        indices = indices.to(device=device, dtype=torch.long)
        cond_vocab = conditioner.token_embedding.weight.shape[0] # cond vocabulary comes from the upper level codebook size

        if torch.any(indices < 0):
            raise ValueError(f"{name} contains negative values")

        # Allow exactly one out-of-range ID for BOS from an upper prior with BOS enabled
        # BOS id is expected to be exactly equal to cond_vocab and is mapped to a neutral token
        if torch.any(indices >= cond_vocab):
            if torch.any(indices > cond_vocab):
                raise ValueError(
                    f"{name} contains values above conditioning vocab (max allowed={cond_vocab})"
                )
            indices = indices.clone()
            indices[indices == cond_vocab] = 0

        # Process the conditioning input through the WaveNet conditioner
        return conditioner(indices)

    @staticmethod
    def _align_cached_embedding(
        embedding: torch.Tensor,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        """!
        @brief Align a cached embedding to the requested batch and sequence length.
        @param embedding Cached embedding tensor shaped `(B, T, D)`.
        @param batch_size Expected batch size for the current request.
        @param seq_len Required sequence length.
        @param device Target device for the embedding.
        @param name Label used in validation errors.
        @return Embedding tensor cropped to `(batch_size, seq_len, D)`.
        @throws ValueError If the embedding shape is incompatible.
        """
        if embedding.ndim != 3:
            raise ValueError(f'{name} must have shape (B, T, D), got {tuple(embedding.shape)}')
        embedding = embedding.to(device=device)
        if embedding.shape[0] == 1 and batch_size > 1:
            embedding = embedding.expand(batch_size, -1, -1)
        if embedding.shape[0] != batch_size:
            raise ValueError(f'{name} batch size {embedding.shape[0]} does not match indices batch size {batch_size}')
        if embedding.shape[1] < seq_len:
            raise ValueError(f'{name} length {embedding.shape[1]} is shorter than requested seq_len={seq_len}')
        return embedding[:, :seq_len, :]

    def _learned_timing_embedding(
        self,
        timing: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """!
        @brief Build bounded learned timing embeddings from song position and duration.
        @param timing Timing tensor shaped `(B, 3)`.
        @param seq_len Sequence length to generate embeddings for.
        @param device Target device for computations.
        @return Timing embedding tensor shaped `(B, seq_len, D)`.
        """
        pos = torch.arange(seq_len, device=device)
        token_time_col = torch.div(pos, self.block_len, rounding_mode='floor')
        window_time_cols = max(1, (self.max_seq_len + self.block_len - 1) // self.block_len)
        rel_denom = max(1, window_time_cols - 1)
        rel_fraction = (token_time_col.float() / rel_denom).clamp(0.0, 1.0)
        rel_idx = torch.round(rel_fraction * (self.timing_num_bins - 1)).long()

        start_s = timing[:, 0:1].clamp_min(0.0)
        duration_s = timing[:, 1:2].clamp_min(1e-6)
        window_s = self.timing_window_seconds
        if window_s is None or window_s <= 0:
            # Fall back to using the stored segment start fraction only.
            abs_fraction = timing[:, 2:3].clamp(0.0, 1.0).expand(-1, seq_len)
        else:
            token_offset_s = rel_fraction.view(1, seq_len) * float(window_s)
            abs_fraction = ((start_s + token_offset_s) / duration_s).clamp(0.0, 1.0)
        abs_idx = torch.round(abs_fraction * (self.timing_num_bins - 1)).long()

        max_duration = max(self.timing_max_duration_seconds, 1.0)
        duration_clamped = duration_s.clamp(1.0, max_duration)
        duration_fraction = torch.log1p(duration_clamped) / torch.log1p(
            torch.tensor(max_duration, device=device, dtype=torch.float32)
        )
        duration_idx = torch.round(
            duration_fraction.clamp(0.0, 1.0) * (self.duration_num_bins - 1)
        ).long().squeeze(1)

        rel_emb = self.relative_timing_embedding(rel_idx).unsqueeze(0)
        abs_emb = self.absolute_timing_embedding(abs_idx)
        dur_emb = self.duration_timing_embedding(duration_idx).unsqueeze(1)
        return abs_emb + rel_emb + dur_emb

    def loss(
        self,
        indices: torch.Tensor,
        upper_indices: Optional[torch.Tensor] = None,
        second_upper_indices: Optional[torch.Tensor] = None,
        timing: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """!
        @brief Compute next-token cross-entropy on a full sequence tensor (B, T).
        @param indices The input token indices for the current level (shape: [batch_size, seq_len]).
        @param upper_indices Upper-level token indices for upsampler priors (shape: [batch_size, upper_seq_len]).
        @param second_upper_indices Second upper-level token indices for dual-conditioning priors.
        @param timing Optional timing tensor (shape: [batch_size, 3]) with
        [start_time_seconds, total_duration_seconds, fraction_elapsed].
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

    @torch.inference_mode()
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
            start_tokens = start_tokens.to(device).long()
            if self.use_bos_token:
                # Continuation prefixes still need the same BOS-shifted positions used during training.
                bos = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
                tokens = torch.cat([bos, start_tokens], dim=1)
                bos_prefix_len = 1
            else:
                tokens = start_tokens
        else:
            if self.use_bos_token:
                tokens = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
                bos_prefix_len = 1
            else:
                tokens = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        
        target_len = seq_len + bos_prefix_len

        if tokens.shape[1] > target_len:
            raise ValueError("start_tokens length cannot exceed requested seq_len")

        cached_conditioning_emb = None
        cached_second_conditioning_emb = None
        if self.is_upsampler:
            cached_conditioning_emb = self._compute_conditioning_embedding(
                upper_indices,
                self.conditioner,
                device,
                'upper_indices',
            )
            if self.second_conditioner is not None:
                cached_second_conditioning_emb = self._compute_conditioning_embedding(
                    second_upper_indices,
                    self.second_conditioner,
                    device,
                    'second_upper_indices',
                )

        cached_timing_emb = None
        if self.use_timing_conditioning and timing is not None:
            timing_for_cache = timing.to(device=device, dtype=torch.float32)
            if timing_for_cache.ndim == 1:
                timing_for_cache = timing_for_cache.unsqueeze(0)
            if timing_for_cache.shape[0] == 1 and batch_size > 1:
                timing_for_cache = timing_for_cache.expand(batch_size, -1)
            if timing_for_cache.shape != (batch_size, 3):
                raise ValueError(
                    f"timing must have shape (B, 3) or (1, 3), got {tuple(timing_for_cache.shape)}"
                )
            cached_timing_emb = self._learned_timing_embedding(timing_for_cache, self.max_seq_len, device)

        while tokens.shape[1] < target_len:
            logits = self.forward(
                tokens,
                upper_indices=None if cached_conditioning_emb is not None else upper_indices,
                second_upper_indices=None if cached_second_conditioning_emb is not None else second_upper_indices,
                timing=None if cached_timing_emb is not None else timing,
                conditioning_emb=cached_conditioning_emb,
                second_conditioning_emb=cached_second_conditioning_emb,
                timing_emb=cached_timing_emb,
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
