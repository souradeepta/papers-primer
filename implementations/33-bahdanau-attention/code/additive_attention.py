"""Bahdanau additive attention over encoder states during decoding."""

from __future__ import annotations

import torch


class AdditiveAttention(torch.nn.Module):
    """Score decoder query against every encoder state, then form context."""

    def __init__(self, width: int, attention_width: int = 12) -> None:
        super().__init__()
        self.key = torch.nn.Linear(width, attention_width, bias=False)
        self.query = torch.nn.Linear(width, attention_width, bias=False)
        self.energy = torch.nn.Linear(attention_width, 1, bias=False)

    def forward(self, query: torch.Tensor, encoder_states: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return soft alignment weights and their weighted encoder context."""
        scores = self.energy(torch.tanh(self.key(encoder_states) + self.query(query)[:, None])).squeeze(-1)
        weights = scores.masked_fill(~valid, float("-inf")).softmax(dim=-1)
        return torch.einsum("bs,bsd->bd", weights, encoder_states), weights


def main() -> None:
    torch.manual_seed(33)
    attention = AdditiveAttention(width=8)
    encoder_states, query = torch.randn(2, 5, 8), torch.randn(2, 8)
    valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    context, weights = attention(query, encoder_states, valid)
    context.square().mean().backward()
    print(f"context: {tuple(context.shape)}; valid alignment sums: {weights.sum(-1).tolist()}")
    assert torch.allclose(weights.sum(-1), torch.ones(2)) and weights[1, 3:].eq(0).all()
    print("ok: decoder queries softly align to valid source positions")


if __name__ == "__main__":
    main()
