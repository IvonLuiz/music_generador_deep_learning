import torch
import torch.nn as nn
import torch.nn.functional as F


class FactoredAttention(nn.Module):
    """!
    
    """
    def __init__(self, model_dim, num_heads, block_len, attention_type: str):
        """!
        @brief Initializes the FactoredAttention module.
        @param model_dim: The dimensionality of the input and output features.
        @param num_heads: The number of attention heads.
        @param block_len: The length of the blocks for factored attention.
        @param attention_type: The type of attention to use ('row', 'column', or 'previous_row').
        """
        super().__init__()
        assert attention_type in ['row', 'column', 'previous_row'], "attention_type must be one of 'row', 'column', or 'previous_row'"
        
        self.num_heads = num_heads
        self.block_len = block_len
        self.attention_type = attention_type

        self.qkv_proj = nn.Linear(model_dim, 3 * model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """!
        @brief Forward pass of the FactoredAttention module.
        @param x: Input tensor of shape (batch_size, seq_len, model_dim).
        @return: Output tensor of shape (batch_size, seq_len, model_dim).
        """
        batch_size, seq_len, model_dim = x.size()
        
        # reshape the 1D sequence into 2D grid of blocks
        num_blocks = seq_len // self.block_len
        x_2d = x.view(batch_size, num_blocks, self.block_len, model_dim)  # (batch_size, num_blocks, block_len, model_dim)
    
        # Project input to Q, K, V
        qkv = self.qkv_proj(x_2d)  # (batch_size, num_blocks, block_len, 3 * model_dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # Each of shape (batch_size, num_blocks, block_len, model_dim)

        # Compute attention scores based on the specified attention type
        if self.attention_type == 'row':
            # ROW ATTENTION: treat 'num_blocks' as part of the batch dimension and attend within each block
            # every block attends internally to its own tokens
            q = q.view(batch_size * num_blocks, self.block_len, model_dim)
            k = k.view(batch_size * num_blocks, self.block_len, model_dim)
            v = v.view(batch_size * num_blocks, self.block_len, model_dim)
            
            # apply standard casual mask here since it's autoregressive within each block
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (batch_size * num_blocks, block_len, model_dim)
            out = out.view(batch_size, num_blocks, self.block_len, model_dim)  # (batch_size, num_blocks, block_len, model_dim)
        
        elif self.attention_type == 'column':
            # COLUMN ATTENTION: We treat 'Block_Len' as part of the batch dimension.
            # Token `i` in block `j` attends to token `i` in blocks `0` to `j-1`.
            q = q.transpose(1, 2).reshape(batch_size * self.block_len, num_blocks, model_dim)
            k = k.transpose(1, 2).reshape(batch_size * self.block_len, num_blocks, model_dim)
            v = v.transpose(1, 2).reshape(batch_size * self.block_len, num_blocks, model_dim)
            
            # Apply causal mask because you can't attend to future blocks
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.view(batch_size, self.block_len, num_blocks, model_dim).transpose(1, 2) # (batch_size, num_blocks, block_len, model_dim)
            
        # flatten back to 1D sequence and project
        out = out.contiguous().view(batch_size, seq_len, model_dim)  # (batch_size, seq_len, model_dim)
        out = self.out_proj(out)  # (batch_size, seq_len, model_dim
        return out


if __name__ == "__main__":
    print("--- Testing FactoredAttention ---")
    BATCH_SIZE = 32
    NUM_HEADS = 4
    MODEL_DIM = 256 # equivalent to embeddings dim (must be divisible by NUM_HEADS)
    BLOCK_LEN = 16
    NUM_BLOCKS = 4
    SEQ_LEN = BLOCK_LEN * NUM_BLOCKS  # 64
    
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
    
    print("\nAll implemented attention patterns passed successfully!")