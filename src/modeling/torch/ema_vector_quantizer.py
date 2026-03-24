import torch
import torch.nn as nn
import torch.nn.functional as F


class EMAVectorQuantizer(nn.Module):
    """
    Exponential Moving Average Vector Quantization layer with straight-through estimator.
    This implementation uses random restarts to prevent codebook collapse.
    """
    def __init__(
        self, 
        num_embeddings: int,
        embedding_dim: int,
        beta: float = 0.25,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
        restart_threshold: float = 0.5
    ):
        """!
        @brief Initializes the EMAVectorQuantizer module.
        @param num_embeddings: K, codebook size.
        @param embedding_dim: D, embedding vector width (channel dimension of latent).
        @param beta: commitment loss weight (typically 0.25 .. 2.0).
        @param ema_decay: exponential moving average decay factor.
        @param epsilon: small constant for numerical stability.
        @param restart_threshold: threshold for triggering random restarts.
        """
        super().__init__()
        # Coerce values coming from YAML configs (which can be loaded as strings).
        self.num_embeddings = int(num_embeddings)  # K
        self.embedding_dim = int(embedding_dim)    # D
        self.beta = float(beta)
        self.decay = float(ema_decay)
        self.epsilon = float(epsilon)
        self.restart_threshold = float(restart_threshold)

        if self.num_embeddings <= 0:
            raise ValueError("num_embeddings must be > 0")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        if not (0.0 < self.decay < 1.0):
            raise ValueError("ema_decay must be in (0, 1)")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")
        if self.restart_threshold < 0.0:
            raise ValueError("restart_threshold must be >= 0")

        # Initialize codebook embeddings
        embedding = torch.empty(num_embeddings, embedding_dim)
        nn.init.uniform_(embedding, -1.0 / num_embeddings, 1.0 / num_embeddings)
        self.register_buffer('embedding', embedding)  # (K, D)
        
        # Buffers for EMA updated
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))  # (K,)
        self.register_buffer('embedding_avg', embedding.clone())  # (K, D)

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

        # Move channel dim to last and flatten spatial dims: (B, D, *spatial) -> (B, *spatial, D) -> (N, D)
        perm = [0] + list(range(2, z_e.dim())) + [1]
        z = z_e.permute(*perm).contiguous()             # (B, *spatial, D)
        z_flat = z.view(-1, D)                          # (N, D)  where N = B * prod(spatial)

        # Compute distances ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z*e
        e = self.embedding                                          # (K, D)
        z_sq = (z_flat ** 2).sum(dim=1, keepdim=True)               # (N, 1)
        e_sq = (e ** 2).sum(dim=1)                                  # (K,)
        distances = z_sq + e_sq.unsqueeze(0) - 2 * (z_flat @ e.t()) # (N, K)

        # Encoding - nearest embedding index per position (nearest neighbor)
        indices = torch.argmin(distances, dim=1)                            # (N,)
        encodings = F.one_hot(indices, self.num_embeddings).type_as(z_flat) # (N, K)

        # Update EMA codebook
        if self.training:
            # Update cluster size (how many times each code was used)
            # N_i^(t) = \gamma * N_i^(t-1) + (1 - \gamma) * n_i^(t)
            # for decay (gamma) 0.99, this means we keep 99% of the previous count and add 1% of the new count
            encodings_sum = encodings.sum(dim=0)  # (K,)
            self.cluster_size.data.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)

            # Update embedding average
            # m_i^(t) = \gamma * m_i^(t-1) + (1 - \gamma) * sum_{j} z_j * 1{e_j = i}
            encodings_weighted_sum = encodings.t() @ z_flat  # (K, D)
            self.embedding_avg.data.mul_(self.decay).add_(encodings_weighted_sum, alpha=1 - self.decay)

            # Random restarts for underutilized codes
            dead_codes_idx = torch.where(self.cluster_size < self.restart_threshold)[0]  # indices of underutilized codes
            if len(dead_codes_idx) > 0:
                # Randomly pick vectors from the current batch's continuous outputs
                random_indices = torch.randint(0, z_flat.shape[0], (len(dead_codes_idx),), device=z_flat.device)
                random_vectors = z_flat[random_indices]  # (num_restarts, D)
                
                # Reset EMA buffers for dead codes
                self.cluster_size.data[dead_codes_idx] = self.restart_threshold
                self.embedding_avg.data[dead_codes_idx] = (random_vectors * self.restart_threshold).to(self.embedding_avg.dtype)
            
            # Normalize the updated codebook with Laplace smoothing
            # formula from DeepMind: (N_i + epsilon) / (sum(N) + K * epsilon)
            n = self.cluster_size.sum()
            cluster_size_smoothed = (
                (self.cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            )
            embed_normalized = self.embedding_avg / cluster_size_smoothed.unsqueeze(1)
            self.embedding.data.copy_(embed_normalized)

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


if __name__ == "__main__":
    print("--- Testing EMAVectorQuantizer ---")
    
    # Setup dummy dimensions
    BATCH_SIZE = 2
    CHANNELS = 64
    NUM_EMBEDDINGS = 512
    H, W = 16, 16
    
    # Initialize module
    vq = EMAVectorQuantizer(
        num_embeddings=NUM_EMBEDDINGS, 
        embedding_dim=CHANNELS, 
        restart_threshold=1.0
    )
    
    # 1. Test Forward Pass Shapes (2D)
    print("\n1. Testing 2D Forward Pass & Shapes...")
    z_2d = torch.randn(BATCH_SIZE, CHANNELS, H, W)
    z_q, indices, vq_loss, cb_loss, c_loss = vq(z_2d)
    
    assert z_q.shape == z_2d.shape, f"z_q shape mismatch: {z_q.shape}"
    assert indices.shape == (BATCH_SIZE, H, W), f"indices shape mismatch: {indices.shape}"
    print("Shapes are correct!")
    
    # 2. Test EMA Updates (Training Mode)
    print("\n2. Testing EMA Buffer Updates...")
    initial_cluster_sum = vq.cluster_size.sum().item()
    assert initial_cluster_sum > 0, "Cluster size should have updated during the forward pass!"
    print(f"Cluster sizes updated! Sum is now: {initial_cluster_sum:.4f}")
    
    # 3. Test Random Restarts
    print("\n3. Testing Random Restart Logic for Dead Codes...")
    # Force the first 5 codes to be "dead" (usage = 0.0)
    vq.cluster_size.data[:5] = 0.0
    dead_codes_before = (vq.cluster_size < 1.0).sum().item()
    print(f"Forced {dead_codes_before} codes to be dead.")
    
    # Run forward pass to trigger restarts
    _ = vq(z_2d)
    dead_codes_after = (vq.cluster_size < 1.0).sum().item()
    
    # Keep in mind: normal EMA decay might push some *other* codes slightly below 1.0 
    # depending on the batch size, but the explicitly dead ones should be revived to 1.0
    assert vq.cluster_size[0].item() >= 1.0, "Code 0 was not restarted!"
    print(f"Restarts triggered successfully! Dead codes revived.")

    # 4. Test Eval Mode (Frozen Codebook)
    print("\n4. Testing Eval Mode (Frozen Buffers)...")
    vq.eval()
    cluster_size_before_eval = vq.cluster_size.clone()
    embed_before_eval = vq.embedding.clone()
    
    # Run forward pass in eval mode
    _ = vq(z_2d)
    
    assert torch.equal(cluster_size_before_eval, vq.cluster_size), "Cluster size changed during eval!"
    assert torch.equal(embed_before_eval, vq.embedding), "Embeddings changed during eval!"
    print("Eval mode correctly froze all EMA buffers!")

    print("\nAll EMAVectorQuantizer tests passed successfully!")
