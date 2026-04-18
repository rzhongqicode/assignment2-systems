import math
import torch
from einops import einsum, rearrange

class FlashAttention2Pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False): 
        # 1. Extract the correct 3D dimensions: [Batch, Seq_Len, Dim]
        B, N_q, dim = Q.shape
        _, N_k, _ = K.shape
        
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        # 2. Initialize tensors with the Batch dimension included
        O = torch.zeros((B, N_q, dim), device=Q.device)
        l = torch.zeros((B, N_q), device=Q.device)
        m = torch.full((B, N_q), fill_value=-math.inf, device=Q.device)
        L = torch.zeros((B, N_q), device=Q.device)

        # Outer loop: Iterate over Q blocks
        for i in range(N_q // Q_TILE_SIZE):
            q_start = i * Q_TILE_SIZE
            q_end = (i + 1) * Q_TILE_SIZE

            # 3. Keep the leading Batch dimension when slicing (using [:, ...])
            Q_tile = Q[:, q_start:q_end, :]  # [B, Q_TILE, d]
            O_i = O[:, q_start:q_end, :]     # [B, Q_TILE, d]
            m_i = m[:, q_start:q_end]        # [B, Q_TILE]
            l_i = l[:, q_start:q_end]        # [B, Q_TILE]

            # Inner loop: Iterate over K, V blocks
            for j in range(N_k // K_TILE_SIZE):
                k_start = j * K_TILE_SIZE
                k_end = (j + 1) * K_TILE_SIZE

                K_tile = K[:, k_start:k_end, :]  # [B, K_TILE, d]
                V_tile = V[:, k_start:k_end, :]  # [B, K_TILE, d]

                # 4. Compute local attention scores (support Batch dim using 'b')
                S_ij = einsum(Q_tile, K_tile, "b q d, b k d -> b q k") / math.sqrt(dim)
                max_s_ij = torch.max(S_ij, dim=-1).values  # [B, Q_TILE]
                
                old_m_i = m_i.clone()
                m_i = torch.maximum(old_m_i, max_s_ij)
                
                # Rearrange while keeping the Batch dimension
                unsqueezed_m_i = rearrange(m_i, "b q -> b q 1")
                P_ij = torch.exp(S_ij - unsqueezed_m_i)
                
                # Compute scaling factor and update the exponential sum (l_i)
                scale = torch.exp(old_m_i - m_i)
                l_i = scale * l_i + torch.sum(P_ij, dim=-1)
                
                # Update output O_i (add 'b' dimension to Einsum operations)
                O_i = einsum(scale, O_i, "b q, b q d -> b q d") + einsum(P_ij, V_tile, "b q k, b k d -> b q d")

            # 5. Write back the computed tile to the global output tensor
            O[:, q_start:q_end, :] = O_i / l_i.unsqueeze(-1)
            m[:, q_start:q_end] = m_i
            l[:, q_start:q_end] = l_i
            
            # Compute LogSumExp (L) to be used in the backward pass
            L[:, q_start:q_end] = m_i + torch.log(l_i)

        # Save tensors required for the backward pass into the context
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError

def cdiv(a, b):
    """Ceiling division"""
    return (a + b - 1) // b