"""Seq2Seq LSTM encoder-decoder with teacher-forced decoding.

The encoder maps a variable-length source sequence to final hidden/cell state;
the decoder starts from that state and predicts target tokens one step at a
time. This is the fixed-vector bottleneck architecture from Sutskever et al.
before attention was added.
"""

from __future__ import annotations

import torch


class Seq2Seq(torch.nn.Module):
    """Small LSTM encoder-decoder for token-id sequences."""

    def __init__(self, vocabulary: int, width: int = 16) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary, width)
        self.encoder = torch.nn.LSTM(width, width, batch_first=True)
        self.decoder = torch.nn.LSTM(width, width, batch_first=True)
        self.output = torch.nn.Linear(width, vocabulary)

    def forward(self, source: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        """Encode source then decode with teacher-forced previous target tokens."""
        _, state = self.encoder(self.embedding(source))
        decoded, _ = self.decoder(self.embedding(decoder_input), state)
        return self.output(decoded)


def main() -> None:
    torch.manual_seed(32)
    model = Seq2Seq(vocabulary=12)
    source = torch.tensor([[4, 5, 6, 2], [7, 8, 2, 0]])
    decoder_input = torch.tensor([[1, 6, 5, 4], [1, 8, 7, 0]])
    target = torch.tensor([[6, 5, 4, 2], [8, 7, 2, 0]])
    logits = model(source, decoder_input)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten())
    loss.backward()
    print(f"decoder logits: {tuple(logits.shape)}; teacher-forced loss: {loss.item():.3f}")
    assert logits.shape == (2, 4, 12) and model.encoder.weight_ih_l0.grad is not None
    print("ok: one final encoder state initializes autoregressive target decoding")


if __name__ == "__main__":
    main()
