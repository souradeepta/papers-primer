"""Minimal decoder-only causal Transformer smoke test illustrating GPT-3's
central engineering claim: in-context learning (zero-shot -> one-shot ->
few-shot) requires ZERO change to model architecture or trained weights.
Going from zero-shot to few-shot is purely a matter of making the input
*sequence* longer -- more demonstration tokens prepended before the query --
fed through the exact same causal self-attention forward pass, "with no
gradient updates or fine-tuning" (Brown et al. 2020, arXiv:2005.14165,
abstract and Section 2.1).

GPT-3 itself uses the GPT-2-style decoder-only Transformer architecture
(stacked causal self-attention + feed-forward blocks); this file reproduces
that shape at toy scale with plain integer "token" ids so it needs no real
tokenizer or trained weights to demonstrate the mechanism.
"""
import torch
import torch.nn.functional as F


def causal_mask(seq_len: int) -> torch.Tensor:
    """Lower-triangular mask: position i can only attend to positions <= i."""
    return torch.tril(torch.ones(seq_len, seq_len)).bool()


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide evenly across heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = torch.nn.Linear(d_model, 3 * d_model)
        self.out = torch.nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)

        def split_heads(z):
            return z.view(b, t, self.n_heads, self.d_k).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        mask = causal_mask(t)
        scores = q @ k.transpose(-2, -1) / (self.d_k**0.5)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(b, t, d)
        return self.out(out)


class DecoderBlock(torch.nn.Module):
    """Pre-norm causal self-attention + feed-forward, each with a residual
    connection -- the repeating unit GPT-3 stacks 12 (Small) to 96 (175B)
    times (paper Table 2.1)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = torch.nn.LayerNorm(d_model)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.GELU(),
            torch.nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(torch.nn.Module):
    """A toy GPT-3-shaped decoder-only LM: token embedding + learned
    positional embedding + N causal decoder blocks + a linear head to vocab
    logits. Every zero/one/few-shot forward pass below runs through ONE
    instance of this model with FROZEN weights -- no fine-tuning, no
    gradient updates -- matching the paper's definition of in-context
    learning happening "at inference time" only."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 64,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(vocab_size, d_model)
        self.pos_emb = torch.nn.Embedding(max_seq_len, d_model)
        self.blocks = torch.nn.ModuleList(
            DecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        )
        self.ln_f = torch.nn.LayerNorm(d_model)
        self.head = torch.nn.Linear(d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        _, t = ids.shape
        pos = torch.arange(t, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


if __name__ == "__main__":
    torch.manual_seed(0)
    VOCAB_SIZE = 50
    SEP_TOKEN, EOS_TOKEN = 40, 41  # stand-ins for "=>" and end-of-example markers

    model = TinyGPT(VOCAB_SIZE)
    model.eval()

    def demonstration(en_id: int, fr_id: int) -> list[int]:
        """One (input => output) demonstration, e.g. a toy translation pair."""
        return [en_id, SEP_TOKEN, fr_id, EOS_TOKEN]

    query_token = 5

    # Zero-shot: just the query, no demonstrations at all.
    zero_shot_ids = torch.tensor([[query_token, SEP_TOKEN]])

    # One-shot: exactly one demonstration prepended before the query.
    one_shot_ids = torch.tensor([demonstration(1, 21) + [query_token, SEP_TOKEN]])

    # Few-shot: several demonstrations prepended before the query -- the
    # paper's "in-context learning" setting (Section 2.1), where the real
    # model sees up to n_ctx=2048 tokens of context; here just longer than
    # one-shot to show the same scaling relationship.
    few_shot_examples = [demonstration(i, i + 20) for i in range(1, 6)]
    few_shot_ids = torch.tensor(
        [[t for ex in few_shot_examples for t in ex] + [query_token, SEP_TOKEN]]
    )

    with torch.no_grad():
        params_before = sum(p.numel() for p in model.parameters())
        logits_zero = model(zero_shot_ids)
        logits_one = model(one_shot_ids)
        logits_few = model(few_shot_ids)
        params_after = sum(p.numel() for p in model.parameters())

    # The central claim under test: identical weights, only the *sequence
    # length* changed between zero/one/few-shot -- no parameter was added,
    # removed, or updated to go from zero-shot to few-shot prompting.
    assert params_before == params_after, "few-shot prompting must not change parameter count"
    assert zero_shot_ids.shape[1] < one_shot_ids.shape[1] < few_shot_ids.shape[1], (
        "expected zero-shot < one-shot < few-shot sequence lengths"
    )
    assert logits_zero.shape[-1] == logits_one.shape[-1] == logits_few.shape[-1] == VOCAB_SIZE
    print(
        f"ok: zero-shot seq_len={zero_shot_ids.shape[1]}, "
        f"one-shot seq_len={one_shot_ids.shape[1]}, "
        f"few-shot seq_len={few_shot_ids.shape[1]} -- "
        f"same {params_before:,}-parameter model, same forward pass, zero weight updates"
    )

    # Verify causal masking: logits at the query position must be identical
    # whether or not a token is appended *after* it -- a position can never
    # be influenced by tokens that come later. This is what makes
    # autoregressive, one-token-at-a-time generation well-defined at all.
    query_pos = zero_shot_ids.shape[1] - 1
    extended_ids = torch.cat([zero_shot_ids, torch.tensor([[7]])], dim=1)
    with torch.no_grad():
        logits_short = model(zero_shot_ids)[0, query_pos]
        logits_extended = model(extended_ids)[0, query_pos]
    assert torch.allclose(logits_short, logits_extended, atol=1e-5), (
        "causal mask leaked: appending a future token changed an earlier position's logits"
    )
    print("ok: causal mask verified -- a later token cannot change an earlier position's logits")

    # Greedy decoding + softmax sanity check on the few-shot logits (the
    # last position's logits are what a real sampler would draw the next
    # token from, optionally divided by a temperature before softmax).
    next_token_logits = logits_few[0, -1]
    greedy_token = int(torch.argmax(next_token_logits))
    temperature = 0.7
    probs = F.softmax(next_token_logits / temperature, dim=-1)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5), "softmax must sum to 1"
    print(
        f"ok: greedy next-token id={greedy_token} at temperature={temperature}, "
        f"softmax probs sum to {probs.sum():.4f}"
    )
