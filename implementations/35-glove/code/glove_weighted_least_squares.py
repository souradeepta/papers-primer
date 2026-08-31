"""GloVe weighted least-squares objective over a toy co-occurrence matrix."""

from __future__ import annotations

import torch


class GloVe(torch.nn.Module):
    """Target/context word vectors and biases used by the GloVe objective."""

    def __init__(self, vocabulary: int, width: int = 6) -> None:
        super().__init__()
        self.word = torch.nn.Embedding(vocabulary, width)
        self.context = torch.nn.Embedding(vocabulary, width)
        self.word_bias = torch.nn.Embedding(vocabulary, 1)
        self.context_bias = torch.nn.Embedding(vocabulary, 1)

    def loss(self, rows: torch.Tensor, columns: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Fit dot product plus biases to log co-occurrence with GloVe weighting."""
        prediction = (self.word(rows) * self.context(columns)).sum(-1)
        prediction = prediction + self.word_bias(rows).squeeze(-1) + self.context_bias(columns).squeeze(-1)
        weights = torch.minimum((counts / 10).pow(.75), torch.ones_like(counts))
        return (weights * (prediction - counts.log()).square()).mean()


def main() -> None:
    torch.manual_seed(35)
    rows = torch.tensor([0, 0, 1, 1, 2, 2])
    columns = torch.tensor([1, 2, 0, 2, 0, 1])
    counts = torch.tensor([20., 5., 20., 8., 5., 8.])
    model = GloVe(3)
    optimizer = torch.optim.Adam(model.parameters(), lr=.05)
    before = model.loss(rows, columns, counts)
    for _ in range(100):
        optimizer.zero_grad(); loss = model.loss(rows, columns, counts); loss.backward(); optimizer.step()
    print(f"weighted log-count loss: {before.item():.3f} -> {loss.item():.3f}")
    assert loss < before and model.word.weight.grad is not None
    print("ok: word/context vectors factorize log co-occurrence statistics")


if __name__ == "__main__":
    main()
