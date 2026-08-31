"""Dropout training versus deterministic inference with inverted scaling."""

from __future__ import annotations

import torch


class DropoutClassifier(torch.nn.Module):
    """MLP that samples thinned subnetworks only while training."""

    def __init__(self, probability: float = .5) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(4, 12), torch.nn.ReLU(), torch.nn.Dropout(probability),
            torch.nn.Linear(12, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def main() -> None:
    torch.manual_seed(34)
    model, data = DropoutClassifier(), torch.randn(16, 4)
    model.train()
    first, second = model(data), model(data)
    training_loss = torch.nn.functional.cross_entropy(first, torch.randint(0, 2, (16,)))
    training_loss.backward()
    model.eval()
    evaluation_first, evaluation_second = model(data), model(data)
    print(f"training outputs differ: {not torch.allclose(first, second)}; eval outputs match: {torch.allclose(evaluation_first, evaluation_second)}")
    assert not torch.allclose(first, second) and torch.allclose(evaluation_first, evaluation_second)
    print("ok: random thinned networks train together while inference is deterministic")


if __name__ == "__main__":
    main()
