"""Minimal bidirectional self-attention + MLM masking, runnable smoke test.

Mirrors two core mechanics of Devlin et al. 2018 (arXiv:1810.04805):

1. Bidirectional self-attention (section 3, "BERT"): unlike a decoder's
   causally-masked self-attention (Vaswani et al. 2017), every token in a
   BERT encoder layer attends to every other token in the sequence with no
   directional mask at all -- left and right context are both visible in
   every layer.
2. The masked-language-model (MLM) input corruption procedure (section
   3.1): for each training sequence, 15% of WordPiece token positions are
   selected; of those, 80% are replaced with [MASK], 10% with a random
   token from the vocabulary, and 10% are left unchanged. The model's job
   is to predict the *original* token at every selected position, using
   both left and right context.

This file re-implements both from scratch (reusing the same scaled
dot-product attention formula as the Transformer paper, since BERT is a
Transformer encoder stack) and asserts properties that would fail if
either mechanic were implemented wrong.
"""

# Reading guide: follow the named helpers in data-flow order, then inspect the
# assertions at the bottom. Change one toy input at a time and rerun the file.
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.

    mask, if given, is a boolean tensor broadcastable to the (..., seq,
    seq) score matrix; True means "allowed to attend", False means
    "blocked". Passing mask=None (BERT's default) means every position
    attends to every other position -- fully bidirectional.
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights


class BidirectionalSelfAttention(torch.nn.Module):
    """Single-head self-attention with no directional mask -- the encoder
    self-attention used in every BERT layer (contrast with a GPT-style
    decoder layer, which would apply a causal mask here)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        out, weights = scaled_dot_product_attention(q, k, v, mask=None)
        return out, weights


def causal_mask(seq_len: int) -> torch.Tensor:
    """Lower-triangular mask, included only as a contrast case: this is
    what a GPT-style decoder layer would apply, and what BERT never
    applies in its encoder."""
    return torch.tril(torch.ones(seq_len, seq_len)).bool()


def apply_mlm_masking(token_ids: torch.Tensor, vocab_size: int, mask_token_id: int,
                       mask_prob: float = 0.15, generator=None):
    """Paper section 3.1's 80/10/10 masking-language-model corruption rule.

    Returns (corrupted_ids, labels) where labels holds the original token
    id at every selected position and -100 (the standard "ignore this
    position" sentinel for cross-entropy) everywhere else.
    """
    labels = token_ids.clone()
    select_mask = torch.rand(token_ids.shape, generator=generator) < mask_prob
    labels[~select_mask] = -100

    corrupted = token_ids.clone()
    rand = torch.rand(token_ids.shape, generator=generator)

    replace_with_mask = select_mask & (rand < 0.8)
    replace_with_random = select_mask & (rand >= 0.8) & (rand < 0.9)
    # remaining 10% (rand >= 0.9 and selected): left unchanged, on purpose.

    corrupted[replace_with_mask] = mask_token_id
    n_random = int(replace_with_random.sum().item())
    if n_random > 0:
        corrupted[replace_with_random] = torch.randint(
            0, vocab_size, (n_random,), generator=generator
        )
    return corrupted, labels, select_mask


if __name__ == "__main__":
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)

    # --- 1. Bidirectional self-attention: every position attends to every
    # other position, including positions *after* it. This is the property
    # that distinguishes a BERT encoder layer from a GPT decoder layer.
    d_model, seq_len, batch = 16, 6, 2
    x = torch.randn(batch, seq_len, d_model)
    layer = BidirectionalSelfAttention(d_model)
    out, weights = layer(x)
    assert out.shape == x.shape, f"expected {x.shape}, got {out.shape}"

    # A fully bidirectional layer must assign nonzero attention weight
    # to at least one position strictly *after* the query position --
    # something a causally-masked decoder can never do by construction.
    upper_triangle = weights.triu(diagonal=1)
    assert upper_triangle.sum() > 0, (
        "expected nonzero attention to future positions in a bidirectional layer"
    )
    print(f"ok: bidirectional attention output shape {tuple(out.shape)} matches input, "
          f"and attends to future positions (sum of future weights = {upper_triangle.sum():.4f})")

    # Contrast case: apply a causal mask to the same scores and confirm it
    # zeroes out exactly the future-position weights that were nonzero above.
    mask = causal_mask(seq_len)
    q, k, v = layer.q_proj(x), layer.k_proj(x), layer.v_proj(x)
    _, causal_weights = scaled_dot_product_attention(q, k, v, mask)
    causal_upper = causal_weights.triu(diagonal=1)
    assert torch.allclose(causal_upper, torch.zeros_like(causal_upper)), (
        "causal mask should zero all attention weight to future positions"
    )
    print("ok: applying a causal mask to the same layer zeroes all future-position "
          "weights -- this is the GPT-style contrast case BERT does not use")

    # --- 2. MLM masking procedure: 15% of positions selected, split
    # 80% [MASK] / 10% random / 10% unchanged, labels only set on selected
    # positions.
    vocab_size, mask_token_id = 1000, 999
    seq_len_mlm = 512  # large enough for the 80/10/10 split ratios to hold approximately
    token_ids = torch.randint(0, vocab_size - 1, (1, seq_len_mlm), generator=gen)
    corrupted, labels, select_mask = apply_mlm_masking(
        token_ids, vocab_size, mask_token_id, mask_prob=0.15, generator=gen
    )

    n_selected = int(select_mask.sum().item())
    frac_selected = n_selected / seq_len_mlm
    assert 0.10 < frac_selected < 0.20, (
        f"expected ~15% of positions selected, got {frac_selected:.3f}"
    )

    n_mask_token = int((corrupted[select_mask] == mask_token_id).sum().item())
    frac_mask_token = n_mask_token / n_selected
    assert 0.65 < frac_mask_token < 0.95, (
        f"expected ~80% of selected positions replaced with [MASK], got {frac_mask_token:.3f}"
    )

    # Every non-selected position must be untouched (same id) and ignored
    # in the loss (label == -100).
    unselected = ~select_mask
    assert torch.equal(corrupted[unselected], token_ids[unselected]), (
        "unselected positions must be left untouched"
    )
    assert (labels[unselected] == -100).all(), (
        "unselected positions must be masked out of the loss with label -100"
    )
    print(f"ok: MLM masking selected {frac_selected:.1%} of {seq_len_mlm} positions "
          f"(target ~15%), {frac_mask_token:.1%} of those replaced with [MASK] (target ~80%)")
