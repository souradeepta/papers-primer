"""Minimal Adam step showing first-step bias correction."""

# Reading guide: follow the named helpers in data-flow order, then inspect the
# assertions at the bottom. Change one toy input at a time and rerun the file.
from __future__ import annotations
import math

def main() -> None:
    gradient, beta1, beta2, lr, epsilon = 2.0, .9, .999, .1, 1e-8
    m = (1-beta1)*gradient; v = (1-beta2)*gradient**2
    m_hat = m/(1-beta1); v_hat = v/(1-beta2)
    parameter = 1.0 - lr*m_hat/(math.sqrt(v_hat)+epsilon)
    print(f"m={m:.3f}, v={v:.3f}, corrected=({m_hat:.3f}, {v_hat:.3f}), parameter={parameter:.3f}")
    assert abs(m_hat-gradient) < 1e-12 and abs(v_hat-gradient**2) < 1e-12
    assert parameter < 1.0
    print("ok: bias correction makes Adam's first moment estimates unbiased")
if __name__ == '__main__': main()
