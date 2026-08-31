"""Compact Vision Transformer: patch embeddings, class token, encoder stack."""
from __future__ import annotations
import torch


class VisionTransformer(torch.nn.Module):
    """Classify an image by treating non-overlapping patches as tokens."""
    def __init__(self, image_size: int = 16, patch: int = 4, width: int = 32) -> None:
        super().__init__()
        count = (image_size // patch) ** 2
        self.patch_embed = torch.nn.Conv2d(3, width, patch, stride=patch)
        self.class_token = torch.nn.Parameter(torch.zeros(1, 1, width))
        self.position = torch.nn.Parameter(torch.zeros(1, count + 1, width))
        layer = torch.nn.TransformerEncoderLayer(width, nhead=4, dim_feedforward=2*width,
                                                  batch_first=True, norm_first=True)
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=2)
        self.head = torch.nn.Linear(width, 3)
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Patch-project then prepend class token before Transformer encoding."""
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls = self.class_token.expand(len(images), -1, -1)
        return self.head(self.encoder(torch.cat([cls, tokens], 1) + self.position)[:, 0])


def main() -> None:
    torch.manual_seed(25); model = VisionTransformer()
    images, labels = torch.randn(4, 3, 16, 16), torch.tensor([0, 1, 2, 1])
    logits = model(images); loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    print(f"logits: {tuple(logits.shape)}; tokens incl. class: 17")
    assert logits.shape == (4, 3) and model.patch_embed.weight.grad is not None
    print("ok: patches, positional embeddings, class token, and encoder form a ViT classifier")


if __name__ == "__main__":
    main()
