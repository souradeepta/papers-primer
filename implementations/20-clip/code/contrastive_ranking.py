"""CLIP's dual encoders and symmetric image-text contrastive objective."""

from __future__ import annotations
import torch


class Encoder(torch.nn.Module):
    """Small modality encoder that returns an embedding vector."""
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(input_size, 8)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.layer(x), dim=-1)


def clip_loss(images: torch.Tensor, texts: torch.Tensor, temperature: float) -> torch.Tensor:
    """Average image-to-text and text-to-image cross entropy over paired rows."""
    logits = images @ texts.T / temperature
    labels = torch.arange(len(images))
    return (torch.nn.functional.cross_entropy(logits, labels) +
            torch.nn.functional.cross_entropy(logits.T, labels)) / 2


def main() -> None:
    torch.manual_seed(20)
    image_encoder, text_encoder = Encoder(5), Encoder(6)
    image_data, text_data = torch.randn(4, 5), torch.randn(4, 6)
    optimizer = torch.optim.Adam([*image_encoder.parameters(), *text_encoder.parameters()], lr=.05)
    before = clip_loss(image_encoder(image_data), text_encoder(text_data), .1)
    for _ in range(80):
        optimizer.zero_grad()
        loss = clip_loss(image_encoder(image_data), text_encoder(text_data), .1)
        loss.backward(); optimizer.step()
    image_embeddings, text_embeddings = image_encoder(image_data), text_encoder(text_data)
    similarities = image_embeddings @ text_embeddings.T
    print(f"symmetric contrastive loss: {before.item():.3f} -> {clip_loss(image_embeddings, text_embeddings, .1).item():.3f}")
    assert torch.all(similarities.argmax(1) == torch.arange(4))
    print("ok: normalized dual encoders learn matching image-text pairs in both directions")


if __name__ == "__main__":
    main()
