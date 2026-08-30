"""Adam optimizer with first/second moments and bias correction.

This compact scalar optimizer follows Algorithm 1 of Kingma and Ba: it
accumulates exponential gradient moments, corrects their early-step bias, and
uses the normalized update to minimize a quadratic objective.
"""

from __future__ import annotations

import math


class Adam:
    """Stateful Adam update for a scalar parameter."""

    def __init__(self, learning_rate: float = .1, beta1: float = .9, beta2: float = .999) -> None:
        self.learning_rate, self.beta1, self.beta2 = learning_rate, beta1, beta2
        self.m = self.v = 0.0
        self.step_count = 0

    def step(self, parameter: float, gradient: float) -> float:
        """Update moments and return one bias-corrected Adam parameter step."""
        self.step_count += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient * gradient
        m_hat = self.m / (1 - self.beta1 ** self.step_count)
        v_hat = self.v / (1 - self.beta2 ** self.step_count)
        return parameter - self.learning_rate * m_hat / (math.sqrt(v_hat) + 1e-8)


def main() -> None:
    optimizer, parameter = Adam(), 5.0
    initial_loss = (parameter - 1.0) ** 2
    for _ in range(80):
        parameter = optimizer.step(parameter, 2 * (parameter - 1.0))
    final_loss = (parameter - 1.0) ** 2
    print(f"quadratic loss: {initial_loss:.3f} -> {final_loss:.6f}")
    assert final_loss < initial_loss * .001 and optimizer.step_count == 80
    print("ok: Adam's corrected moments minimize the objective")


if __name__ == "__main__":
    main()
