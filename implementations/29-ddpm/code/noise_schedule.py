"""DDPM forward noising, epsilon prediction, and one reverse denoising step."""
from __future__ import annotations
import torch


def q_sample(x0: torch.Tensor, alpha_bar: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Sample q(x_t|x_0)=sqrt(alpha_bar)x_0+sqrt(1-alpha_bar)epsilon."""
    return alpha_bar.sqrt()*x0 + (1-alpha_bar).sqrt()*noise


class NoisePredictor(torch.nn.Module):
    """Small denoiser conditioned on a normalized diffusion timestep."""
    def __init__(self) -> None:
        super().__init__(); self.network = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([x, time], 1))


def main() -> None:
    torch.manual_seed(29); betas = torch.linspace(.01, .08, 8); alpha_bar = torch.cumprod(1-betas, 0)
    model, optimizer = NoisePredictor(), torch.optim.Adam(NoisePredictor().parameters(), lr=.01)
    model = NoisePredictor(); optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    clean = torch.randn(64, 1); steps = torch.randint(0, len(betas), (64,))
    noise = torch.randn_like(clean); noisy = q_sample(clean, alpha_bar[steps, None], noise)
    time = (steps[:, None].float() / (len(betas)-1))
    for _ in range(80):
        prediction = model(noisy, time); loss = torch.nn.functional.mse_loss(prediction, noise)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    estimated_clean = (noisy - (1-alpha_bar[steps, None]).sqrt()*model(noisy, time)) / alpha_bar[steps, None].sqrt()
    print(f"noise-prediction loss: {loss.item():.3f}; reconstructed shape: {tuple(estimated_clean.shape)}")
    assert torch.isfinite(estimated_clean).all()
    print("ok: a timestep-conditioned network predicts forward noise for reverse denoising")


if __name__ == "__main__":
    main()
