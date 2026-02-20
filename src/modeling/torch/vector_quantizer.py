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
        z_e: (B, D, L) for 1D (audio) or (B, D, H, W) for 2D (spectrograms/images).
        Returns:
            z_q_st: quantized tensor (straight-through), same shape as z_e
            indices_map: selected code indices — (B, L) for 1D or (B, H, W) for 2D
            vq_loss: VQ loss (codebook_loss + beta * commitment_loss)
            codebook_loss: ||sg[z_e] - e||^2
            commitment_loss: beta * ||z_e - sg[z_q]||^2
        """
        # --- Shape handling: supports both 1D (B, D, L) and 2D (B, D, H, W) ---
        original_shape = z_e.shape
        B, D = original_shape[0], original_shape[1]
        spatial_dims = original_shape[2:]  # (L,) for 1D, (H, W) for 2D

        assert D == self.embedding_dim, f"Expected channels==embedding_dim ({self.embedding_dim}), got {D}"

        # Move channel dim to last and flatten spatial dims: (B, D, *spatial) -> (N, D)
        # permute: (B, *spatial, D)
        perm = [0] + list(range(2, z_e.dim())) + [1]
        z = z_e.permute(*perm).contiguous()             # (B, *spatial, D)
        z_flat = z.view(-1, D)                          # (N, D)  where N = B * prod(spatial)

        # Compute distances ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z.e
        e = self.embedding                                          # (K, D)
        z_sq = (z_flat ** 2).sum(dim=1, keepdim=True)               # (N, 1)
        e_sq = (e ** 2).sum(dim=1)                                  # (K,)
        distances = z_sq + e_sq.unsqueeze(0) - 2 * (z_flat @ e.t()) # (N, K)

        # Encoding - nearest embedding index per position (nearest neighbor)
        indices = torch.argmin(distances, dim=1)                            # (N,)
        encodings = F.one_hot(indices, self.num_embeddings).type_as(z_flat) # (N, K)

        # Quantize and reshape back to original shape
        z_q_flat = encodings @ e                                # (N, D)
        z_q = z_q_flat.view(*([B] + list(spatial_dims) + [D]))  # (B, *spatial, D)

        # Permute back: (B, *spatial, D) -> (B, D, *spatial)
        inv_perm = [0, z_e.dim() - 1] + list(range(1, z_e.dim() - 1))
        z_q = z_q.permute(*inv_perm).contiguous()   # (B, D, *spatial)

        # Losses
        ## Codebook loss: ||sg[z_e] - e||^2
        ## gradients flow only to codebook embeddings (z_q)
        codebook_loss = F.mse_loss(z_e.detach(), z_q)
        ## Commitment loss: ||z_e - sg[z_q]||^2
        ## gradients flow only to encoder (z_e)
        commitment_loss = F.mse_loss(z_e, z_q.detach())

        vq_loss = codebook_loss + self.beta * commitment_loss

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        # Reshape indices to (B, *spatial)
        indices_map = indices.view(*([B] + list(spatial_dims)))

        return z_q_st, indices_map, vq_loss, codebook_loss, self.beta * commitment_loss
