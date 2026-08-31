"""Bahdanau additive attention with cached encoder projections and padding masks.

The decoder supplies one query per output step. Each query is compared against
all encoder states, normalized over real source positions, and used to retrieve
a weighted context vector. This is the alignment computation in the paper.
"""

from __future__ import annotations

import torch


class AdditiveAttention(torch.nn.Module):
    """Score decoder query against every encoder state, then form context."""

    def __init__(self, width: int, attention_width: int = 12) -> None:
        super().__init__()
        self.key = torch.nn.Linear(width, attention_width, bias=False)
        self.query = torch.nn.Linear(width, attention_width, bias=False)
        self.energy = torch.nn.Linear(attention_width, 1, bias=False)

    def project_encoder(self, encoder_states: torch.Tensor) -> torch.Tensor:
        """Project source states once; reuse this result across decoder steps."""
        return self.key(encoder_states)

    def forward(
        self, query: torch.Tensor, encoder_states: torch.Tensor, valid: torch.Tensor,
        projected_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked alignment weights and the matching weighted context."""
        keys = self.project_encoder(encoder_states) if projected_states is None else projected_states
        scores = self.energy(torch.tanh(keys + self.query(query)[:, None])).squeeze(-1)
        weights = scores.masked_fill(~valid, float("-inf")).softmax(dim=-1)
        return torch.einsum("bs,bsd->bd", weights, encoder_states), weights


def main() -> None:
    torch.manual_seed(33)
    attention = AdditiveAttention(width=8)
    encoder_states, query = torch.randn(2, 5, 8), torch.randn(2, 8)
    valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    cached_keys = attention.project_encoder(encoder_states)
    # A decoder would repeat this call once for every generated target token.
    context, weights = attention(query, encoder_states, valid, cached_keys)
    second_context, second_weights = attention(query + .1, encoder_states, valid, cached_keys)
    (context.square().mean() + second_context.square().mean()).backward()
    print(f"context={tuple(context.shape)}; alignment sums={weights.sum(-1).tolist()}")
    assert torch.allclose(weights.sum(-1), torch.ones(2)) and weights[1, 3:].eq(0).all()
    assert second_weights[1, 3:].eq(0).all() and attention.key.weight.grad is not None
    print("ok: decoder queries softly align to valid source positions")


if __name__ == "__main__":
    main()
