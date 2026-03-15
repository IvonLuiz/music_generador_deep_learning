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
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads # dimension per head
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
        num_blocks = seq_len // self.block_len

        # project input to Q, K, V
        qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3 * model_dim)
        # reshape into 2d grid and split into attention heads
        qkv = qkv.view(batch_size, num_blocks, self.block_len, 3, self.num_heads, self.head_dim) # (batch_size, num_blocks, block_len, 3, num_heads, head_dim)
        # split into q, k, v
        q, k, v = qkv.unbind(dim=3)  # Each of shape (batch_size, num_blocks, block_len, num_heads, head_dim)

        # Compute attention scores based on the specified attention type
        # jukebox paper: 
        #   Row attention attends within the block_len dimension, while column attention attends across the num_blocks dimension. 
        #   Both types of attention are causal, meaning that they cannot attend to future positions.
        if self.attention_type == 'row':
            # ROW ATTENTION: Attend within the block_len
            # treat 'num_blocks' as part of the batch dimension and attend within each block
            # SDPA expects [Batch, Heads, Seq, Head_Dim]
            # merge `B` and `num_blocks` into the batch dimension
            q = q.transpose(2,3).reshape(batch_size * num_blocks, self.num_heads, self.block_len, self.head_dim)
            k = k.transpose(2,3).reshape(batch_size * num_blocks, self.num_heads, self.block_len, self.head_dim)
            v = v.transpose(2,3).reshape(batch_size * num_blocks, self.num_heads, self.block_len, self.head_dim)

            # SDPA thinks it is processing Batch * Num_Blocks completely separate sequences of length Block_Len. The blocks cannot see each other.
            # apply standard casual mask here since it's autoregressive within each block
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (batch_size * num_blocks, block_len, model_dim)
            out = out.transpose(1,2).reshape(batch_size, num_blocks, self.block_len, model_dim)  # (batch_size, num_blocks, block_len, model_dim)
        
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
            
    
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True) # apply causal mask because you can't attend to future blocks
            # reshape back to the 2D block grid
            out = out.reshape(batch_size, self.block_len, self.num_heads, num_blocks, self.head_dim)
            out = out.transpose(1, 2).reshape(batch_size, num_blocks, self.block_len, model_dim)  # (batch_size, num_blocks, block_len, model_dim)

        elif self.attention_type == 'previous_row':
            raise NotImplementedError("previous_row attention is not yet implemented!")

        # Flatten back to 1D sequence and project
        out = out.reshape(batch_size, seq_len, model_dim)   # (batch_size, seq_len, model_dim)
        return self.out_proj(out)   # (batch_size, seq_len, model_dim)


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