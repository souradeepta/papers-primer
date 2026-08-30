"""ResNet basic blocks with identity and projection shortcuts.

Residual learning asks a convolutional branch to predict a correction F(x)
while a shortcut carries x directly. This compact PyTorch implementation
includes the two-convolution basic block, batch normalization, and the
one-by-one projection used when a stage changes channel count or stride.
"""

from __future__ import annotations

import torch


class BasicBlock(torch.nn.Module):
    """The two 3x3 convolution residual block used by ResNet-18/34."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.residual = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels), torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (torch.nn.Identity() if in_channels == out_channels and stride == 1
                         else torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                                                  torch.nn.BatchNorm2d(out_channels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add correction and shortcut before the block's final nonlinearity."""
        return torch.relu(self.residual(x) + self.shortcut(x))


def main() -> None:
    torch.manual_seed(18)
    x = torch.randn(2, 8, 16, 16)
    identity = BasicBlock(8, 8)(x)
    downsampled = BasicBlock(8, 16, stride=2)(x)
    print(f"identity stage: {tuple(identity.shape)}; projection stage: {tuple(downsampled.shape)}")
    assert identity.shape == x.shape and downsampled.shape == (2, 16, 8, 8)
    print("ok: residual and projection shortcuts preserve valid stage shapes")


if __name__ == "__main__":
    main()
