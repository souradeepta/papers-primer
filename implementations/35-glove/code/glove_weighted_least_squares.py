"""GloVe from sparse, distance-weighted co-occurrence counts to learned vectors.

The paper trains on observed word-context pairs rather than constructing a
huge dense vocabulary-square matrix. This example builds that sparse training
set, applies the published weighting shape, and optimizes the log-count loss.
"""

from __future__ import annotations

from collections import defaultdict

import torch


def cooccurrence_pairs(tokens: list[int], window: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build observed center-context counts with inverse-distance contribution."""
    counts: defaultdict[tuple[int, int], float] = defaultdict(float)
    for center, word in enumerate(tokens):
        left, right = max(0, center - window), min(len(tokens), center + window + 1)
        for context in range(left, right):
            if context != center:
                counts[word, tokens[context]] += 1.0 / abs(context - center)
    pairs = sorted(counts.items())
    rows = torch.tensor([pair[0][0] for pair in pairs])
    columns = torch.tensor([pair[0][1] for pair in pairs])
    values = torch.tensor([value for _, value in pairs])
    return rows, columns, values


class GloVe(torch.nn.Module):
    """Target/context word vectors and biases used by the GloVe objective."""

    def __init__(self, vocabulary: int, width: int = 6) -> None:
        super().__init__()
        self.word = torch.nn.Embedding(vocabulary, width)
        self.context = torch.nn.Embedding(vocabulary, width)
        self.word_bias = torch.nn.Embedding(vocabulary, 1)
        self.context_bias = torch.nn.Embedding(vocabulary, 1)

    def loss(self, rows: torch.Tensor, columns: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Fit dot products plus biases to weighted observed log co-occurrences."""
        prediction = (self.word(rows) * self.context(columns)).sum(-1)
        prediction = prediction + self.word_bias(rows).squeeze(-1) + self.context_bias(columns).squeeze(-1)
        x_max, alpha = 10.0, .75
        weights = torch.minimum((counts / x_max).pow(alpha), torch.ones_like(counts))
        return (weights * (prediction - counts.log()).square()).mean()


def main() -> None:
    torch.manual_seed(35)
    # A repeated mini-corpus gives a nontrivial sparse co-occurrence table.
    corpus = [0, 1, 2, 1, 0, 2, 3, 2, 0, 1, 3, 1]
    rows, columns, counts = cooccurrence_pairs(corpus, window=2)
    model = GloVe(vocabulary=4, width=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=.05)
    before = model.loss(rows, columns, counts)
    for _ in range(250):
        optimizer.zero_grad(); loss = model.loss(rows, columns, counts); loss.backward(); optimizer.step()
    print(f"sparse pairs={len(counts)}; weighted log-count loss: {before.item():.3f} -> {loss.item():.3f}")
    assert loss < before and model.word.weight.grad is not None
    print("ok: word/context vectors factorize log co-occurrence statistics")


if __name__ == "__main__":
    main()
