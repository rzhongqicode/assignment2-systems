import math

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore
from einops import einsum, rearrange


class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_casual=False):
        dim = Q.shape[-1]
        N_q = Q.shape[0]
        N_k = K.shape[0]
        # ctx.Q_TILE_SIZE = 16
        Q_TILE_SIZE = 16
        # ctx.K_TILE_SIZE = 16
        K_TILE_SIZE = 16

        O = torch.zeros(size=(N_q, dim), device=Q.device)
        l = torch.zeros(size=(N_q), device=Q.device)
        m = torch.full(size=(N_q), fill_value=-math.inf, device=Q.device)
        L = torch.zeros(size=(N_q), device=Q.device)

        for i in range(N_q // Q_TILE_SIZE):
            q_start = i * Q_TILE_SIZE
            q_end = (i + 1) * Q_TILE_SIZE

            Q_tile = Q[q_start:q_end, :]  # [Q_TILE, d]
            O_i = O[q_start:q_end, :]  # [Q_TILE, d]
            m_i = m[q_start:q_end]  # [Q_TILE]
            l_i = l[q_start:q_end]  # [Q_TILE]

            # O_tile = O[i*Q_TILE_SIZE:end_idx,:]
            for j in range(N_k // K_TILE_SIZE):
                k_start = j * K_TILE_SIZE
                k_end = (j + 1) * K_TILE_SIZE

                K_tile = K[k_start:k_end, :]  # [K_TILE, d]
                V_tile = V[k_start:k_end, :]  # [K_TILE, d]

                S_ij = einsum(Q_tile, K_tile, "Bq d, Bk d -> Bq Bk") / math.sqrt(dim)
                max_s_ij = torch.max(S_ij, dim=-1).values  # [Q_TILE]
                old_m_i = m_i.clone()
                m_i = torch.maximum(old_m_i, max_s_ij)
                unsqueezed_m_i = rearrange(m_i, "Q_TILE -> Q_TILE 1")
                P_ij = torch.exp(S_ij - unsqueezed_m_i)
                scale = torch.exp(old_m_i - m_i)
                l_i = scale * l_i + torch.sum(P_ij, dim=-1)
                O_i = einsum(scale, O_i, "Bq, Bq d -> Bq d") + einsum(P_ij, V_tile, "Bq Bk, Bk d -> Bq d")

            O[q_start:q_end, :] = O_i / l_i.unsqueeze(-1)
            m[q_start:q_end] = m_i
            l[q_start:q_end] = l_i
            L[q_start:q_end] = m_i + torch.log(l_i)

        ctx.save_for_backward(Q, K, V, O, L)
        return O

    def backward(ctx, grad_out):
        raise NotImplementedError


def cdiv(a, b):
    return (a + b - 1) // b
