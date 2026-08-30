"""Demonstrate the VAE reparameterization invariant with scalar values."""
from __future__ import annotations
import math

def sample(mu: float, logvar: float, epsilon: float) -> float:
    return mu + math.exp(.5 * logvar) * epsilon

def main() -> None:
    mu, logvar, epsilon = 1.5, math.log(4.0), -0.25
    z = sample(mu, logvar, epsilon)
    print(f'z={z:.3f}; expected={mu + 2*epsilon:.3f}')
    assert abs(z - (mu + 2*epsilon)) < 1e-12
    print('ok: randomness is isolated in epsilon, leaving a differentiable path')
if __name__ == '__main__': main()
