"""Dropout training versus deterministic inference with inverted scaling.

The program exposes the two operational modes, verifies their expected
behavior, and performs a real classifier update so masks participate in
backpropagation instead of being a disconnected random-output demonstration.
"""

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
        """Return class logits; Dropout changes behavior with train and eval mode."""
        return self.layers(x)


def main() -> None:
    torch.manual_seed(34)
    model, data = DropoutClassifier(probability=.4), torch.randn(32, 4)
    labels = (data[:, 0] + data[:, 1] > 0).long()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    model.train()
    first, second = model(data), model(data)
    training_loss = torch.nn.functional.cross_entropy(first, labels)
    optimizer.zero_grad()
    training_loss.backward()
    optimizer.step()
    model.eval()
    evaluation_first, evaluation_second = model(data), model(data)
    stochastic = not torch.allclose(first, second)
    deterministic = torch.allclose(evaluation_first, evaluation_second)
    print(f"loss={training_loss.item():.3f}; training differs={stochastic}; eval matches={deterministic}")
    assert stochastic and deterministic and model.layers[0].weight.grad is not None
    print("ok: random thinned networks train together while inference is deterministic")


if __name__ == "__main__":
    main()
