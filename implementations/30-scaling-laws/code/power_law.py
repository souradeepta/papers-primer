"""Fit and extrapolate a Kaplan-style power-law loss curve.

Scaling laws model held-out cross-entropy as L(C)=a*C^b+c, where compute C
grows and b is negative. This compact example fits the log-linear special
case with c=0 to synthetic measured runs, then predicts an unseen larger
compute budget. Real studies fit several terms and require far more data.
"""

from __future__ import annotations

import math


def fit_power_law(compute: list[float], loss: list[float]) -> tuple[float, float]:
    """Least-squares fit log(loss)=log(a)+b*log(compute)."""
    x, y = [math.log(v) for v in compute], [math.log(v) for v in loss]
    x_mean, y_mean = sum(x) / len(x), sum(y) / len(y)
    exponent = sum((a-x_mean)*(b-y_mean) for a, b in zip(x, y)) / sum((a-x_mean)**2 for a in x)
    return math.exp(y_mean - exponent*x_mean), exponent


def main() -> None:
    compute = [1, 2, 4, 8, 16]
    observed = [2.0, 1.74, 1.52, 1.32, 1.15]
    coefficient, exponent = fit_power_law(compute[:-1], observed[:-1])
    held_out = coefficient * compute[-1] ** exponent
    print(f"fit: L(C)={coefficient:.3f}*C^{exponent:.3f}; held-out prediction={held_out:.3f}")
    assert exponent < 0 and abs(held_out-observed[-1]) < .1
    print("ok: log-space regression captures diminishing loss improvements")


if __name__ == "__main__":
    main()
