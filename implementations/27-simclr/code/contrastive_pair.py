"""SimCLR augmentations, projection head, and NT-Xent contrastive loss."""
from __future__ import annotations
import torch


class SimCLR(torch.nn.Module):
    """Tiny encoder plus nonlinear projection head used only for contrastive loss."""
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU())
        self.projector = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 8))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.projector(self.encoder(x)), dim=-1)


def nt_xent(left: torch.Tensor, right: torch.Tensor, temperature: float = .2) -> torch.Tensor:
    """Classify the matching augmented view among all views in the batch."""
    logits = left @ right.T / temperature
    labels = torch.arange(len(left))
    return (torch.nn.functional.cross_entropy(logits, labels) + torch.nn.functional.cross_entropy(logits.T, labels)) / 2


def main() -> None:
    torch.manual_seed(27); model = SimCLR(); source = torch.randn(8, 8)
    # Small noise stands in for two independent crop/color augmentations.
    left, right = source + .05*torch.randn_like(source), source + .05*torch.randn_like(source)
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    before = nt_xent(model(left), model(right))
    for _ in range(80):
        optimizer.zero_grad(); loss = nt_xent(model(left), model(right)); loss.backward(); optimizer.step()
    similarities = model(left) @ model(right).T
    print(f"NT-Xent: {before.item():.3f} -> {nt_xent(model(left), model(right)).item():.3f}")
    assert torch.all(similarities.argmax(1) == torch.arange(8))
    print("ok: paired augmentations become nearest neighbors in projection space")


if __name__ == "__main__":
    main()
