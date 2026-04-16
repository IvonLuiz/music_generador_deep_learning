import torch
import torch.nn as nn

try:
    # When importing as a module from elsewhere in the project
    from modeling.torch.factored_attention import FactoredAttention
except ImportError:
    # When running this script directly
    from factored_attention import FactoredAttention


class FactoredTransformerLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        block_len: int,
        attention_type: str,
        mlp_dim: int,
        dropout: float,
        qkv_ratio: float = 1.0
    ):
        super().__init__()
        
        # Factored Attention Branch
        self.norm1 = nn.LayerNorm(model_dim)
        self.attn = FactoredAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            block_len=block_len,
            attention_type=attention_type,
            qkv_ratio=qkv_ratio,
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # MLP Branch
        self.norm2 = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, model_dim),
        )
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """!
        @brief Forward pass for the Pre-LN Factored Transformer Layer.
        @param x: Input tensor of shape (batch_size, seq_len, model_dim)
        @return: Output tensor of shape (batch_size, seq_len, model_dim)
        """
        # Attention block with residual connection
        attn_out = self.attn(self.norm1(x))  # (batch_size, seq_len, model_dim)
        x = x + self.dropout1(attn_out)  # Residual connection
        
        # MLP block with residual connection
        mlp_out = self.mlp(self.norm2(x))  # (batch_size, seq_len, model_dim)
        x = x + self.dropout2(mlp_out)  # Residual connection
        
        return x

if __name__ == "__main__":
    print("--- Testing FactoredAttention ---")
    BATCH_SIZE = 32
    NUM_HEADS = 4
    MODEL_DIM = 256 # equivalent to embeddings dim (must be divisible by NUM_HEADS)
    BLOCK_LEN = 16
    NUM_BLOCKS = 4
    SEQ_LEN = BLOCK_LEN * NUM_BLOCKS  # 64
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1
    
    dummy_x = torch.randn(BATCH_SIZE, SEQ_LEN, MODEL_DIM)
    
    for attention_type in ['row', 'column']:
        print(f"\nTesting FactoredAttention ({attention_type})...")
        layer = FactoredTransformerLayer(
            model_dim=MODEL_DIM, 
            num_heads=NUM_HEADS, 
            mlp_dim=DIM_FEEDFORWARD, 
            block_len=BLOCK_LEN, 
            attention_type='column',
            dropout=DROPOUT
        )
    
        out_attn = layer(dummy_x)
        assert out_attn.shape == (BATCH_SIZE, SEQ_LEN, MODEL_DIM), f"{attention_type} attention shape mismatch!"
        print(f"{attention_type.capitalize()} Attention Output Shape: {out_attn.shape} - PASSED")
