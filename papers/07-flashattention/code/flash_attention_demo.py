"""Tiled, online-softmax attention (CPU-only, plain PyTorch).

This reimplements the numerical core of FlashAttention's Algorithm 1 (Dao,
Fu, Ermon, Rudra, Re, 2022, https://arxiv.org/abs/2205.14135): stream over
blocks of keys/values, keep a running row-max m_i and running row-sum l_i,
and rescale the partial output accumulator whenever a new block raises the
running max ("online softmax"). It does NOT implement the paper's actual
IO-aware CUDA kernel, SRAM-sized tiling, or fused backward pass -- there is
no custom kernel here and no GPU is required. The point is to demonstrate
that the tiled, block-streaming algorithm is mathematically exact: it must
produce the same output as ordinary, fully-materialized softmax attention,
just without ever holding the full N x N score matrix in memory at once.
"""

# Reading guide: follow the named helpers in data-flow order, then inspect the
# assertions at the bottom. Change one toy input at a time and rerun the file.
from __future__ import annotations

import torch


def standard_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Reference implementation: materialize the full (Nq, Nk) score matrix.

    This is exactly the quadratic-memory bottleneck FlashAttention avoids:
    `scores` below is an explicit Nq x Nk tensor that must be written to and
    read back from GPU HBM in a naive implementation.
    """
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / (d ** 0.5)   # (Nq, Nk) -- the O(N^2) matrix
    probs = torch.softmax(scores, dim=-1)
    return probs @ v


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size_q: int,
    block_size_kv: int,
) -> torch.Tensor:
    """Tiled attention with online softmax; never materializes the full N x N matrix.

    Mirrors Algorithm 1 of the FlashAttention paper: for each block of
    queries, stream over blocks of keys/values, maintaining a running row
    max `m_i` and running normalizer `l_i`. Whenever a new key/value block
    raises the running max, the previously accumulated output and
    normalizer are rescaled by `exp(m_old - m_new)` so the running sum
    stays a mathematically valid softmax normalizer at every step, not just
    at the end. In the real kernel, `block_size_q`/`block_size_kv` (the
    paper's `Br`/`Bc`) are chosen from the GPU's SRAM size `M` via
    `Bc = ceil(M / (4d))`, `Br = min(ceil(M / (4d)), d)`; here they are
    passed in directly since there is no real SRAM to size against.
    """
    n_q, d = q.shape
    n_k = k.shape[0]
    scale = d ** -0.5
    out = torch.zeros_like(q)

    for qi in range(0, n_q, block_size_q):
        q_blk = q[qi : qi + block_size_q]
        rows = q_blk.shape[0]
        m_i = torch.full((rows, 1), float("-inf"))   # running row max
        l_i = torch.zeros((rows, 1))                   # running row normalizer
        o_i = torch.zeros((rows, d))                    # running (unnormalized) output

        for kj in range(0, n_k, block_size_kv):
            k_blk = k[kj : kj + block_size_kv]
            v_blk = v[kj : kj + block_size_kv]

            s_ij = q_blk @ k_blk.transpose(-2, -1) * scale      # (rows, block_size_kv)
            m_ij = s_ij.max(dim=-1, keepdim=True).values         # this block's row max
            p_ij = torch.exp(s_ij - m_ij)                         # numerically stable exp
            l_ij = p_ij.sum(dim=-1, keepdim=True)

            m_new = torch.maximum(m_i, m_ij)
            # Correction factors: how much the *old* accumulated stats and
            # the *new* block's stats must be rescaled to share one
            # consistent reference max, m_new.
            alpha = torch.exp(m_i - m_new)
            beta = torch.exp(m_ij - m_new)

            l_i = alpha * l_i + beta * l_ij
            o_i = alpha * o_i + beta * (p_ij @ v_blk)
            m_i = m_new

        out[qi : qi + block_size_q] = o_i / l_i

    return out


def main() -> None:
    torch.manual_seed(0)
    n, d = 37, 16   # deliberately not multiples of the block sizes below
    q = torch.randn(n, d)
    k = torch.randn(n, d)
    v = torch.randn(n, d)

    reference = standard_attention(q, k, v)
    tiled = flash_attention(q, k, v, block_size_q=8, block_size_kv=5)

    max_abs_diff = (reference - tiled).abs().max().item()
    print(f"N={n}, d={d}, block_size_q=8, block_size_kv=5")
    print(f"max abs difference between standard and tiled attention: {max_abs_diff:.2e}")

    assert torch.allclose(reference, tiled, atol=1e-5), (
        "tiled online-softmax attention drifted from full-materialization attention"
    )
    print("ok: tiled online-softmax attention exactly matches standard attention (atol=1e-5)")


if __name__ == "__main__":
    main()
