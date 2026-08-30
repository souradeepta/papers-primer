"""A dependency-free, one-step skip-gram negative-sampling demonstration."""
from __future__ import annotations
import math


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def update_dot(dot: float, label: int, learning_rate: float) -> tuple[float, float]:
    """Return updated dot product and binary logistic loss for one pair."""
    probability = sigmoid(dot)
    gradient = label - probability
    loss = -(label * math.log(probability) + (1 - label) * math.log(1 - probability))
    return dot + learning_rate * gradient, loss


def main() -> None:
    positive_after, pos_loss = update_dot(0.0, label=1, learning_rate=0.4)
    negative_after, neg_loss = update_dot(0.0, label=0, learning_rate=0.4)
    print(f"positive dot: 0.000 -> {positive_after:.3f}; loss={pos_loss:.3f}")
    print(f"negative dot: 0.000 -> {negative_after:.3f}; loss={neg_loss:.3f}")
    assert positive_after > 0 and negative_after < 0
    assert pos_loss > 0 and neg_loss > 0
    print("ok: observed pairs are pulled together and sampled noise is pushed apart")


if __name__ == "__main__":
    main()
