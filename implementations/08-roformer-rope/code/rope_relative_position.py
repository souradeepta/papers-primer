"""RoPE inside a compact multi-head attention computation.

RoFormer applies a position-dependent two-dimensional rotation to every
adjacent pair of query and key features. This program implements the
vectorized rotation used by modern Transformer code, feeds the rotated Q/K
tensors through attention, and checks the paper's relative-position identity.
It is CPU-runnable and omits only projection layers and long-context variants.
"""

from __future__ import annotations

import math

import torch


def rotary_angles(sequence_length: int, head_dim: int, base: float = 10_000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the cosine/sine tables for RoPE's geometric frequency schedule."""
    if head_dim % 2:
        raise ValueError("RoPE requires an even attention head dimension")
    positions = torch.arange(sequence_length, dtype=torch.float32)
    frequencies = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    angles = torch.outer(positions, frequencies)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate Q or K shaped (batch, heads, tokens, head_dim) in feature pairs."""
    # Position tables broadcast across batches and heads, but not token rows.
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.empty_like(x)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = even * sin + odd * cos
    return rotated


def causal_rope_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply RoPE then ordinary causal scaled-dot-product attention."""
    _, _, tokens, head_dim = q.shape
    cos, sin = rotary_angles(tokens, head_dim)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
    causal = torch.tril(torch.ones(tokens, tokens, dtype=torch.bool))
    scores.masked_fill_(~causal, float("-inf"))
    return scores.softmax(dim=-1) @ v


def rotate_vector(vector: torch.Tensor, position: int) -> torch.Tensor:
    """Convenience wrapper for the single-vector relative-offset test."""
    cos, sin = rotary_angles(position + 1, vector.numel())
    return apply_rope(vector.view(1, 1, 1, -1), cos[position:position + 1], sin[position:position + 1]).flatten()


def main() -> None:
    torch.manual_seed(8)
    batch, heads, tokens, head_dim = 2, 3, 7, 8
    q = torch.randn(batch, heads, tokens, head_dim)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    output = causal_rope_attention(q, k, v)

    # Shift both positions by three: their relative offset remains unchanged.
    vector_q, vector_k = q[0, 0, 4], k[0, 0, 1]
    original = torch.dot(rotate_vector(vector_q, 4), rotate_vector(vector_k, 1))
    shifted = torch.dot(rotate_vector(vector_q, 7), rotate_vector(vector_k, 4))
    print(f"attention output shape: {tuple(output.shape)}")
    print(f"same relative-offset scores: {original:.6f}, {shifted:.6f}")
    assert output.shape == (batch, heads, tokens, head_dim)
    assert torch.allclose(original, shifted, atol=1e-6)
    assert torch.allclose(rotate_vector(vector_q, 7).norm(), vector_q.norm(), atol=1e-5)
    print("ok: RoPE preserves norms and makes attention scores depend on relative positions")


if __name__ == "__main__":
    main()
