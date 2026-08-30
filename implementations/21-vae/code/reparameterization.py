"""A compact variational autoencoder with an ELBO training step."""

from __future__ import annotations
import torch
import torch.nn.functional as functional


class VAE(torch.nn.Module):
    """Encode 2D observations into a Gaussian latent and decode them."""
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(2, 12), torch.nn.ReLU())
        self.mean, self.log_variance = torch.nn.Linear(12, 2), torch.nn.Linear(12, 2)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(2, 12), torch.nn.ReLU(), torch.nn.Linear(12, 2))
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x); mean, logvar = self.mean(hidden), self.log_variance(hidden)
        z = mean + torch.exp(.5 * logvar) * torch.randn_like(mean)
        return self.decoder(z), mean, logvar


def elbo_loss(x: torch.Tensor, reconstruction: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Negative ELBO: reconstruction error plus KL to N(0,I)."""
    reconstruction_loss = functional.mse_loss(reconstruction, x)
    kl = -.5 * torch.mean(1 + logvar - mean.square() - logvar.exp())
    return reconstruction_loss + kl


def main() -> None:
    torch.manual_seed(21); model = VAE(); data = torch.randn(64, 2) + torch.tensor([1., -1.])
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    initial = None
    for _ in range(100):
        reconstruction, mean, logvar = model(data); loss = elbo_loss(data, reconstruction, mean, logvar)
        if initial is None: initial = loss.item()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    print(f"negative ELBO: {initial:.3f} -> {loss.item():.3f}")
    assert loss.item() < initial
    print("ok: reparameterization supports gradient training of reconstruction and KL terms")


if __name__ == "__main__":
    main()
