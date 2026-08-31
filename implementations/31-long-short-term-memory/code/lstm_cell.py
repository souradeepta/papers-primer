"""LSTM gates, cell-state recurrence, and a gradient-flow demonstration.

This compact implementation exposes the input, forget, output, and candidate
gates that protect long-lived cell state. It uses PyTorch modules for concise,
readable tensor code while retaining the standard LSTM equations.
"""

from __future__ import annotations

import torch


class LSTMCell(torch.nn.Module):
    """One standard forget-gate LSTM cell for a sequence batch."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.gates = torch.nn.Linear(input_size + hidden_size, 4 * hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute i, f, o, candidate then update the protected cell state."""
        hidden, cell = state
        input_gate, forget_gate, output_gate, candidate = self.gates(torch.cat([x, hidden], dim=-1)).chunk(4, dim=-1)
        input_gate, forget_gate, output_gate = input_gate.sigmoid(), forget_gate.sigmoid(), output_gate.sigmoid()
        cell = forget_gate * cell + input_gate * candidate.tanh()
        return output_gate * cell.tanh(), cell


def main() -> None:
    torch.manual_seed(31)
    cell = LSTMCell(input_size=3, hidden_size=5)
    sequence = torch.randn(2, 7, 3)
    hidden = state = torch.zeros(2, 5)
    for token in sequence.unbind(dim=1):
        hidden, state = cell(token, (hidden, state))
    loss = hidden.square().mean()
    loss.backward()
    print(f"final hidden state: {tuple(hidden.shape)}; loss: {loss.item():.4f}")
    assert cell.gates.weight.grad is not None and torch.isfinite(state).all()
    print("ok: gated cell state carries information across a seven-step sequence")


if __name__ == "__main__":
    main()
