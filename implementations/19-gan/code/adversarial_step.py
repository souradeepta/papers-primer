"""A compact GAN with alternating discriminator and generator optimization.

This program implements the minimax game's practical non-saturating variant:
the discriminator classifies real data and generated samples, then the
generator updates through the frozen discriminator to make fakes look real.
The distribution is one-dimensional so the full training loop is CPU-runnable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional


class Generator(torch.nn.Module):
    """Map Gaussian noise to a scalar synthetic data point."""

    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.network(noise)


class Discriminator(torch.nn.Module):
    """Return a real-data logit for a scalar input."""

    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.LeakyReLU(.2), torch.nn.Linear(16, 1))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.network(data)


def main() -> None:
    torch.manual_seed(19)
    generator, discriminator = Generator(), Discriminator()
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=.02)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=.02)

    for _ in range(120):
        real = torch.randn(64, 1) * .5 + 2.0
        noise = torch.randn(64, 1)
        fake = generator(noise)
        # Maximize log D(real)+log(1-D(fake)) by minimizing BCE labels.
        discriminator_loss = (functional.binary_cross_entropy_with_logits(discriminator(real), torch.ones_like(real))
                              + functional.binary_cross_entropy_with_logits(discriminator(fake.detach()), torch.zeros_like(fake)))
        discriminator_optimizer.zero_grad(); discriminator_loss.backward(); discriminator_optimizer.step()

        fake = generator(noise)
        # Non-saturating generator loss: maximize log D(G(z)).
        generator_loss = functional.binary_cross_entropy_with_logits(discriminator(fake), torch.ones_like(fake))
        generator_optimizer.zero_grad(); generator_loss.backward(); generator_optimizer.step()

    generated_mean = generator(torch.randn(512, 1)).mean().item()
    print(f"generated mean after alternating updates: {generated_mean:.3f}")
    assert 0.5 < generated_mean < 3.5
    print("ok: generator and discriminator perform alternating adversarial updates")


if __name__ == "__main__":
    main()
