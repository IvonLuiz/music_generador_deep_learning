import torch
import torch.nn as nn
import torch.nn.functional as F


class FactoredAttention(nn.Module):
    """!
    
    """
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        block_len: int,
        attention_type: str,
        qkv_ratio: float = 1.0
    ):
        """!
        @brief Initializes the FactoredAttention module.
        @param model_dim: The dimensionality of the input and output features.
        @param num_heads: The number of attention heads.
        @param block_len: The length of the blocks for factored attention.
        @param attention_type: The type of attention to use ('row', 'column', or 'previous_row').
        @param qkv_ratio: The ratio of the model dimension to use for Q, K, V projections. Default is 1.0 (i.e., no reduction).
        """
        super().__init__()
        assert attention_type in ['row', 'column', 'previous_row'], "attention_type must be one of 'row', 'column', or 'previous_row'"
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        assert qkv_ratio > 0, "qkv_ratio must be > 0"

        self.num_heads = num_heads
        self.qkv_ratio = float(qkv_ratio)
        self.qkv_dim_total = max(num_heads, int(round(model_dim * self.qkv_ratio)))
        if self.qkv_dim_total % num_heads != 0:
            self.qkv_dim_total = ((self.qkv_dim_total + num_heads - 1) // num_heads) * num_heads
        self.head_dim = self.qkv_dim_total // num_heads
        self.block_len = block_len
        self.attention_type = attention_type

        self.qkv_proj = nn.Linear(model_dim, 3 * self.qkv_dim_total)
        self.out_proj = nn.Linear(self.qkv_dim_total, model_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """!
        @brief Forward pass of the FactoredAttention module.
        @param x: Input tensor of shape (batch_size, seq_len, model_dim).
        @return: Output tensor of shape (batch_size, seq_len, model_dim).
        """
        batch_size, seq_len, model_dim = x.size() # seq_len is the amount of tokens in the sequence (block_len * num_blocks)
        num_blocks = seq_len // self.block_len
        
        assert self.qkv_dim_total == self.num_heads * self.head_dim, "qkv_dim_total must be equal to num_heads * head_dim"
        assert seq_len % self.block_len == 0, "seq_len must be divisible by block_len"

        # project input to Q, K, V
        qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3 * qkv_dim_total)
        # reshape into 2d grid and split into attention heads
        qkv = qkv.view(batch_size, num_blocks, self.block_len, 3, self.num_heads, self.head_dim) # (batch_size, num_blocks, block_len, 3, num_heads, head_dim)
        # split into q, k, v
        q, k, v = qkv.unbind(dim=3)  # Each of shape (batch_size, num_blocks, block_len, num_heads, head_dim)

        # Compute attention scores based on the specified attention type
        # torch scaled_dot_product_attention expects q, k, v to be in shape (Batch, Heads, Seq, Head_Dim), so we will
        # need to reshape/permute accordingly for each attention type.
        # jukebox paper:
        #   Row attention attends within the block_len dimension, while column attention attends across the num_blocks dimension. 
        #   Both types of attention are causal, meaning that they cannot attend to future positions.
        #   Previous-row attention is a special case where each row attends entirely to the previous row, which can be
        #   implemented by shifting the K and V tensors down by one row and applying a causal mask that prevents attending to the current row.
        if self.attention_type == 'row':
            # ROW ATTENTION: Attend within the block_len
            # treat 'num_blocks' as part of the batch dimension and attend within each block
            # SDPA expects [Batch, Heads, Seq, Head_Dim]
            # merge `B` and `num_blocks` into the batch dimension
            q = q.transpose(2,3).reshape(batch_size * num_blocks, self.num_heads, self.block_len, self.head_dim)
            k = k.transpose(2,3).reshape(batch_size * num_blocks, self.num_heads, self.block_len, self.head_dim)
            v = v.transpose(2,3).reshape(batch_size * num_blocks, self.num_heads, self.block_len, self.head_dim)

            # SDPA thinks it is processing Batch * Num_Blocks completely separate sequences of length Block_Len. The blocks cannot see each other
            # apply standard casual mask here since it's autoregressive within each block
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # (batch_size * num_blocks, num_heads, block_len, head_dim)
            out = out.transpose(1,2).reshape(batch_size, num_blocks, self.block_len, self.qkv_dim_total) # (batch_size, num_blocks, block_len, qkv_dim_total)
        
        elif self.attention_type == 'column':
            # COLUMN ATTENTION: Attend across num_blocks
            # treat 'Block_Len' as part of the batch dimension.
            # merge `B` and `block_len` into the batch dimension.
            q = q.permute(0, 2, 3, 1, 4).reshape(
                batch_size * self.block_len, self.num_heads, num_blocks, self.head_dim
            )  # (batch_size * block_len, num_heads, num_blocks, head_dim)
            k = k.permute(0, 2, 3, 1, 4).reshape(
                batch_size * self.block_len, self.num_heads, num_blocks, self.head_dim
            )
            v = v.permute(0, 2, 3, 1, 4).reshape(
                batch_size * self.block_len, self.num_heads, num_blocks, self.head_dim
            )

            # SDPA thinks it is processing Batch * Block_Len completely separate sequences of length Num_Blocks
            # the positions within the block can see each other, but they cannot see other positions in the same block
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True) # apply causal mask because you can't attend to future blocks
            # reshape back to the 2D block grid
            out = out.reshape(batch_size, self.block_len, self.num_heads, num_blocks, self.head_dim)
            out = out.permute(0, 3, 1, 2, 4).reshape(batch_size, num_blocks, self.block_len, self.qkv_dim_total)

        elif self.attention_type == 'previous_row':
            # PREVIOUS-ROW ATTENTION: Attend entirely to the row immediately above.
            # Q comes from the current row. K and V come from the previous row.
            
            # We want to shift K and V down by one row along the `num_blocks` dimension (dim=1), so that each row attends to the previous row
            # The first row will attend to an all-zero row (since there is no previous row), and the last row will be dropped (since there is no next row to attend to it)
            previous_row_zeros = torch.zeros_like(k[:, :1, ...]) # getting right shape from slicing the first row

            # Shift K and V down by 1 row along the `num_blocks` dimension (dim=1)
            # We concatenate the zeros at the top, and slice off the last row
            k_shifted = torch.cat([previous_row_zeros, k[:, :-1, ...]], dim=1)
            v_shifted = torch.cat([previous_row_zeros, v[:, :-1, ...]], dim=1)
            
            # Apply the exact same reshaping as Row Attention
            # Merge B and num_blocks so SDPA processes each row pairing independently
            q = q.transpose(2,3).reshape(batch_size * num_blocks, self.num_heads, self.block_len, self.head_dim)
            k_shifted = k_shifted.transpose(2,3).reshape(batch_size * num_blocks, self.num_heads, self.block_len, self.head_dim)
            v_shifted = v_shifted.transpose(2,3).reshape(batch_size * num_blocks, self.num_heads, self.block_len, self.head_dim)
            
            # SDPA Execution
            # is_causal=False because the entire previous row is in the past, so it is "fully visible"
            out = F.scaled_dot_product_attention(q, k_shifted, v_shifted, is_causal=False)
            out = out.transpose(1,2).reshape(batch_size, num_blocks, self.block_len, self.qkv_dim_total)

        # Flatten back to 1D sequence and project
        out = out.reshape(batch_size, seq_len, self.qkv_dim_total)
        return self.out_proj(out)   # (batch_size, seq_len, model_dim)


if __name__ == "__main__":
    print("--- Testing FactoredAttention ---")
    BATCH_SIZE = 32
    NUM_HEADS = 4
    MODEL_DIM = 256 # equivalent to embeddings dim (must be divisible by NUM_HEADS)
    BLOCK_LEN = 16
    NUM_BLOCKS = 4
    SEQ_LEN = BLOCK_LEN * NUM_BLOCKS  # 64
    print(f"Batch Size: {BATCH_SIZE}, Num Heads: {NUM_HEADS}, Model Dim: {MODEL_DIM}, Block Len: {BLOCK_LEN}, Seq Len: {SEQ_LEN}")
    
    dummy_x = torch.randn(BATCH_SIZE, SEQ_LEN, MODEL_DIM)
    
    print("Testing Row Attention...")
    row_attn = FactoredAttention(MODEL_DIM, NUM_HEADS, BLOCK_LEN, 'row')
    out_row = row_attn(dummy_x)
    assert out_row.shape == (BATCH_SIZE, SEQ_LEN, MODEL_DIM), "Row attention shape mismatch!"
    print(f"Row Attention Output Shape: {out_row.shape} - PASSED")
    
    print("\nTesting Column Attention...")
    col_attn = FactoredAttention(MODEL_DIM, NUM_HEADS, BLOCK_LEN, 'column')
    out_col = col_attn(dummy_x)
    assert out_col.shape == (BATCH_SIZE, SEQ_LEN, MODEL_DIM), "Column attention shape mismatch!"
    print(f"Column Attention Output Shape: {out_col.shape} - PASSED")
    
    print("\nTesting Previous-Row Attention...")
    prev_row_attn = FactoredAttention(MODEL_DIM, NUM_HEADS, BLOCK_LEN, 'previous_row')
    out_prev_row = prev_row_attn(dummy_x)
    assert out_prev_row.shape == (BATCH_SIZE, SEQ_LEN, MODEL_DIM), "Previous-row attention shape mismatch!"
    print(f"Previous-Row Attention Output Shape: {out_prev_row.shape} - PASSED")

    print("\nAll implemented attention patterns passed successfully!")