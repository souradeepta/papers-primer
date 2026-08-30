"""Tiny scalar GAN objective directions; no ML framework required."""
from __future__ import annotations
import math
def sigmoid(x: float) -> float: return 1/(1+math.exp(-x))
def main() -> None:
    d_real, d_fake = sigmoid(1.0), sigmoid(-1.0)
    discriminator_real_gradient = 1-d_real
    discriminator_fake_gradient = -d_fake
    generator_non_saturating_gradient = 1-d_fake
    print(f"D(real)={d_real:.3f}, D(fake)={d_fake:.3f}")
    assert discriminator_real_gradient > 0 and discriminator_fake_gradient < 0
    assert generator_non_saturating_gradient > 0
    print('ok: D raises real scores, lowers fake scores; G raises fake scores')
if __name__ == '__main__': main()
