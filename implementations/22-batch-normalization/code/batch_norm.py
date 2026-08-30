"""Batch Normalization with train-time batch statistics and running estimates.

BatchNorm normalizes each feature channel over a mini-batch, then restores
learnable scale and bias. During inference it uses exponentially averaged
training statistics instead of the current request, avoiding predictions that
depend on which examples happen to share a batch.
"""

from __future__ import annotations

import torch


class BatchNorm:
    """Minimal channel-wise BatchNorm for two-dimensional batches."""

    def __init__(self, features: int, momentum: float = .1, epsilon: float = 1e-5) -> None:
        self.gamma, self.beta = torch.ones(features), torch.zeros(features)
        self.running_mean, self.running_var = torch.zeros(features), torch.ones(features)
        self.momentum, self.epsilon = momentum, epsilon

    def __call__(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Normalize by fresh batch stats in training or accumulated stats in eval."""
        if training:
            mean, variance = x.mean(0), x.var(0, unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * variance
        else:
            mean, variance = self.running_mean, self.running_var
        return self.gamma * (x - mean) / torch.sqrt(variance + self.epsilon) + self.beta


def main() -> None:
    norm = BatchNorm(features=2)
    train_batch = torch.tensor([[1., 10.], [3., 14.], [5., 18.], [7., 22.]])
    normalized = norm(train_batch, training=True)
    evaluation = norm(torch.tensor([[100., -100.]]), training=False)
    print(f"running mean: {norm.running_mean.tolist()}; eval shape: {tuple(evaluation.shape)}")
    assert torch.allclose(normalized.mean(0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(normalized.var(0, unbiased=False), torch.ones(2), atol=1e-4)
    assert not torch.allclose(evaluation, torch.zeros_like(evaluation))
    print("ok: training uses batch statistics while evaluation uses running estimates")


if __name__ == "__main__":
    main()
