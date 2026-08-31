"""LSTM encoder-decoder translation with padding, teacher forcing, and decoding.

This is the fixed-vector Seq2Seq design from Sutskever, Vinyals, and Le. The
encoder final hidden and cell state initialize the decoder; no attention is
used, so the fixed-vector bottleneck is visible in the API.
"""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pack_padded_sequence


class Seq2Seq(torch.nn.Module):
    """Token-id encoder-decoder with a shared embedding and LSTM state handoff."""

    def __init__(self, vocabulary: int, width: int = 24, pad_id: int = 0) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embedding = torch.nn.Embedding(vocabulary, width, padding_idx=pad_id)
        self.encoder = torch.nn.LSTM(width, width, batch_first=True, num_layers=2)
        self.decoder = torch.nn.LSTM(width, width, batch_first=True, num_layers=2)
        self.output = torch.nn.Linear(width, vocabulary)

    def encode(self, source: torch.Tensor, source_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Ignore source padding and return the final two-layer LSTM state."""
        packed = pack_padded_sequence(
            self.embedding(source), source_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, state = self.encoder(packed)
        return state

    def forward(self, source: torch.Tensor, source_lengths: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        """Teacher-force shifted target tokens and return vocabulary logits."""
        state = self.encode(source, source_lengths)
        decoded, _ = self.decoder(self.embedding(decoder_input), state)
        return self.output(decoded)

    @torch.no_grad()
    def greedy_decode(self, source: torch.Tensor, lengths: torch.Tensor, bos_id: int, eos_id: int, steps: int) -> torch.Tensor:
        """Autoregressively feed each prediction back, unlike teacher-forced training."""
        hidden, cell = self.encode(source, lengths)
        token = torch.full((source.size(0), 1), bos_id, device=source.device)
        generated = []
        for _ in range(steps):
            decoded, (hidden, cell) = self.decoder(self.embedding(token), (hidden, cell))
            token = self.output(decoded[:, -1]).argmax(dim=-1, keepdim=True)
            generated.append(token)
            if token.eq(eos_id).all():
                break
        return torch.cat(generated, dim=1)


def main() -> None:
    """Train one teacher-forced update, then demonstrate free-running decoding."""
    torch.manual_seed(32)
    pad, bos, eos = 0, 1, 2
    model = Seq2Seq(vocabulary=14, pad_id=pad)
    source = torch.tensor([[4, 5, 6, eos], [7, 8, eos, pad]])
    lengths = torch.tensor([4, 3])
    decoder_input = torch.tensor([[bos, 6, 5, 4], [bos, 8, 7, pad]])
    target = torch.tensor([[6, 5, 4, eos], [8, 7, eos, pad]])
    logits = model(source, lengths, decoder_input)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=pad)
    loss.backward()
    generated = model.greedy_decode(source, lengths, bos, eos, steps=5)
    print(f"teacher-forced loss={loss.item():.3f}; free-running tokens={generated.tolist()}")
    assert logits.shape == (2, 4, 14)
    assert model.encoder.weight_ih_l0.grad is not None
    print("ok: one encoder state initializes a variable-length autoregressive decoder")


if __name__ == "__main__":
    main()
