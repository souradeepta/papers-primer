"""A compact U-Net encoder, decoder, skip connections, and segmentation loss."""

from __future__ import annotations
import torch


class UNet(torch.nn.Module):
    """Two-resolution U-Net preserving fine detail with a skip concatenation."""
    def __init__(self) -> None:
        super().__init__()
        self.encode = torch.nn.Sequential(torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.ReLU())
        self.pool = torch.nn.MaxPool2d(2)
        self.bottleneck = torch.nn.Sequential(torch.nn.Conv2d(8, 16, 3, padding=1), torch.nn.ReLU())
        self.up = torch.nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.decode = torch.nn.Conv2d(16, 1, 3, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.encode(x)
        context = self.bottleneck(self.pool(skip))
        restored = self.up(context)
        return self.decode(torch.cat([skip, restored], dim=1))


def main() -> None:
    torch.manual_seed(23); model = UNet()
    image, mask = torch.randn(3, 1, 32, 32), torch.randint(0, 2, (3, 1, 32, 32)).float()
    logits = model(image); loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, mask)
    loss.backward()
    print(f"segmentation logits: {tuple(logits.shape)}; loss: {loss.item():.3f}")
    assert logits.shape == mask.shape and model.encode[0].weight.grad is not None
    print("ok: decoder combines coarse context with encoder detail for dense prediction")


if __name__ == "__main__":
    main()
