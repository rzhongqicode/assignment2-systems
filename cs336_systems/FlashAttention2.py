import math

import torch
import triton
import triton.language as tl
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
            O_i = O[:, q_start:q_end, :]  # [B, Q_TILE, d]
            m_i = m[:, q_start:q_end]  # [B, Q_TILE]
            l_i = l[:, q_start:q_end]  # [B, Q_TILE]

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


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # -----------------------------------------------------------
    # 1. Program Identifiers
    # -----------------------------------------------------------
    # Each program instance handles a specific Q tile within a specific batch.
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # -----------------------------------------------------------
    # 2. Block Pointers Initialization
    # -----------------------------------------------------------
    # We use block pointers to let Triton handle boundary checks and memory coalescing automatically.
    # Base pointers are offset to point to the start of the current batch.

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),  # Row-major: dimension 1 (D) is contiguous in memory
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),  # K and V start at column 0 for the inner loop
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # -----------------------------------------------------------
    # 3. Load Query Tile and Initialize Accumulators
    # -----------------------------------------------------------
    # Q remains stationary for this specific thread block throughout the loop.
    Q_tile = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")  # Shape: [Q_TILE_SIZE, D]

    # Initialize running maximum (m), exponential sum (l), and output accumulator (O).
    # ALL accumulators MUST be float32 to prevent underflow/overflow during summation.
    m = tl.full(shape=(Q_TILE_SIZE,), value=float("-inf"), dtype=tl.float32)
    l_tile = tl.zeros(shape=(Q_TILE_SIZE,), dtype=tl.float32)
    O_tile = tl.zeros(shape=(Q_TILE_SIZE, D), dtype=tl.float32)

    # -----------------------------------------------------------
    # 4. Inner Loop over K and V Tiles
    # -----------------------------------------------------------
    for _ in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # Load current K and V tiles
        K_tile = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")  # [K_TILE_SIZE, D]
        V_tile = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")  # [K_TILE_SIZE, D]

        # Compute pre-softmax attention scores: S = (Q @ K^T) * scale
        K_tile_T = tl.trans(K_tile)
        S = tl.dot(Q_tile, K_tile_T)
        S = S * scale

        # FlashAttention Online Softmax Logic
        max_s = tl.max(S, axis=1)
        old_m = m
        m = tl.maximum(max_s, old_m)

        # Compute exponential of scores minus the CURRENT running maximum
        P = tl.exp(S - m[:, None])

        # Compute scaling factor to discount historical values
        scale_factors = tl.exp(old_m - m)

        # Update exponential sum (l)
        l_tile = scale_factors * l_tile + tl.sum(P, axis=1)

        # Downcast P to match V's dtype before the matrix multiplication
        P = P.to(V_tile.dtype)

        # Update output accumulator: scale historical O_tile and add new weighted values
        O_tile = scale_factors[:, None] * O_tile + tl.dot(P, V_tile)

        # Advance K and V block pointers to the next tile along the sequence dimension
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # -----------------------------------------------------------
    # 5. Epilogue: Final Scaling and Store
    # -----------------------------------------------------------
    # Divide the accumulated outputs by the final exponential sum to complete the Softmax
    O_tile = O_tile / l_tile[:, None]

    # Compute the LogSumExp (L) to be saved for the backward pass
    L = m + tl.log(l_tile)

    # Downcast O_tile back to the target global memory dtype (e.g., float16)
    O_tile = O_tile.to(O_block_ptr.type.element_ty)

    # Store the results back to HBM (Global Memory)
    tl.store(O_block_ptr, O_tile, boundary_check=(0,))
    tl.store(L_block_ptr, L, boundary_check=(0,))


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # make sure that inputs are contiguous on physical memory
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        batch_size, N_q, dim = Q.shape
        _, N_k, _ = K.shape

        O = torch.zeros(size=(batch_size, N_q, dim), dtype=torch.float32, device=Q.device)
        L = torch.zeros(size=(batch_size, N_q), dtype=torch.float32, device=Q.device)
        scale = 1 / math.sqrt(dim)
        ctx.D = dim
        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16
        flash_fwd_kernel[(cdiv(N_q, ctx.Q_TILE_SIZE), batch_size)](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            N_q,
            N_k,
            scale,
            D=ctx.D,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SZIE,
        )
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, grad_outputs):
        raise NotImplementedError
