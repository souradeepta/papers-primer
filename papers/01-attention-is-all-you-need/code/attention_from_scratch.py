"""Minimal scaled dot-product + multi-head attention, runnable smoke test.

Mirrors section 3.2 of Vaswani et al. 2017 (arXiv:1706.03762): a single
scaled dot-product attention function, and a multi-head wrapper that
projects into h parallel heads, applies attention independently per head,
then concatenates and re-projects.
"""
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V   (paper eq. 1)"""
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights


class MultiHeadAttention(torch.nn.Module):
    """h parallel attention heads, each with its own learned Q/K/V projection."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide evenly across heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch, seq_len, d_model = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        out, _ = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(out)


def causal_mask(seq_len: int) -> torch.Tensor:
    """Lower-triangular mask so position i cannot attend to positions > i."""
    return torch.tril(torch.ones(seq_len, seq_len)).bool()


if __name__ == "__main__":
    torch.manual_seed(0)

    # Paper's base-model shape ratios (d_model=512, h=8 -> d_k=64), scaled
    # down here so the smoke test runs in milliseconds.
    x = torch.randn(2, 5, 16)
    mha = MultiHeadAttention(d_model=16, n_heads=4)
    out = mha(x)
    assert out.shape == x.shape, f"expected {x.shape}, got {out.shape}"
    print(f"ok: unmasked output shape {tuple(out.shape)} matches input shape")

    # Causal (decoder self-attention) case: verify masking actually zeroes
    # out attention to future positions.
    mask = causal_mask(5)
    q = k = v = x.view(2, 4, 5, 4)[:, 0]  # reuse x reshaped to a single head, 4 dims
    _, weights = scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), mask)
    upper_triangle = weights.squeeze(1).triu(diagonal=1)
    assert torch.allclose(upper_triangle, torch.zeros_like(upper_triangle)), (
        "causal mask leaked attention to future positions"
    )
    print("ok: causal mask zeroes all attention weights to future positions")
