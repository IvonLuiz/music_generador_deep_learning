import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer with straight-through estimator.

    Args:
        num_embeddings (int): K, codebook size.
        embedding_dim (int): D, embedding vector width (channel dimension of latent).
        beta (float): commitment loss weight (typically 0.25 .. 2.0).
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings  # K
        self.embedding_dim = embedding_dim    # D
        self.beta = beta

        embedding = torch.empty(num_embeddings, embedding_dim)
        nn.init.uniform_(embedding, -1.0 / num_embeddings, 1.0 / num_embeddings)
        self.embedding = nn.Parameter(embedding)  # (K, D)

    def forward(self, z_e: torch.Tensor):
        """
        z_e: (B, D, H, W)
        Returns:
            z_q: quantized tensor with same shape as z_e
            indices: (B, H, W) selected code indices
            loss: VQ loss (codebook + commitment)
        """
        B, D, H, W = z_e.shape
        assert D == self.embedding_dim, f"Expected channels==embedding_dim ({self.embedding_dim}), got {D}"

        # Flatten to (N, D)
        z = z_e.permute(0, 2, 3, 1).contiguous()  # (B, H, W, D)
        z_flat = z.view(-1, D)  # (B*H*W, D)

        # Compute distances with squared L2-norm (Euclidean distance)
        ## ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z.e
        e = self.embedding  # (K, D)
        z_sq = (z_flat ** 2).sum(dim=1, keepdim=True)  # (N, 1)
        e_sq = (e ** 2).sum(dim=1)                     # (K,)
        distances = z_sq + e_sq.unsqueeze(0) - 2 * (z_flat @ e.t())  # (N, K)

        # Encoding - nearest embedding index per position (nearest neighbor)
        indices = torch.argmin(distances, dim=1)  # (N,)
        encodings = F.one_hot(indices, self.num_embeddings).type_as(z_flat)  # (N, K) q(z=k|x)

        # Quantize
        z_q_flat = encodings @ e  # (N, D)
        z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()  # un-flattens (B, D, H, W)

        # Losses
        ## Codebook loss: ||sg[z_e] - e||^2
        codebook_loss = F.mse_loss(z_e.detach(), z_q)
        ## Commitment loss: ||z_e - sg[z_q]||^2
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        # Reshape indices to (B, H, W)
        indices_map = indices.view(B, H, W)

        return z_q_st, indices_map, vq_loss
