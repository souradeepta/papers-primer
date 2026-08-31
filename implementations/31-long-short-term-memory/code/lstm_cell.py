"""A transparent LSTM with gate diagnostics, masking, and gradient checks.

The implementation follows the standard forget-gate LSTM recurrence. It uses
one combined gate projection so readers can connect each tensor operation to
the equations while seeing how a batched padded sequence is handled.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GateValues:
    """Values used at one step for inspecting learned memory behavior."""

    input: torch.Tensor
    forget: torch.Tensor
    output: torch.Tensor
    candidate: torch.Tensor


class LSTMCell(torch.nn.Module):
    """One batched LSTM cell with explicit input, forget, output, and candidate gates."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.gates = torch.nn.Linear(input_size + hidden_size, 4 * hidden_size)
        # Starting forget bias positive gives an initial path for retention.
        with torch.no_grad():
            self.gates.bias[hidden_size : 2 * hidden_size].fill_(1.0)

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], GateValues]:
        """Advance one step and return the next state plus gate diagnostics."""
        hidden, cell = state
        packed = self.gates(torch.cat((x, hidden), dim=-1))
        input_gate, forget_gate, output_gate, candidate = packed.chunk(4, dim=-1)
        gates = GateValues(
            input=input_gate.sigmoid(),
            forget=forget_gate.sigmoid(),
            output=output_gate.sigmoid(),
            candidate=candidate.tanh(),
        )
        next_cell = gates.forget * cell + gates.input * gates.candidate
        next_hidden = gates.output * next_cell.tanh()
        return (next_hidden, next_cell), gates


def run_sequence(
    cell: LSTMCell, tokens: torch.Tensor, lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, list[GateValues]]:
    """Run padded batch-time-feature tokens without updating past each length."""
    batch, steps, _ = tokens.shape
    hidden = cell.gates.weight.new_zeros(batch, cell.hidden_size)
    memory = hidden.clone()
    diagnostics: list[GateValues] = []
    for step in range(steps):
        (next_hidden, next_memory), gates = cell(tokens[:, step], (hidden, memory))
        # A padding vector is still data; preserve state after real sequence end.
        active = (step < lengths).unsqueeze(-1)
        hidden = torch.where(active, next_hidden, hidden)
        memory = torch.where(active, next_memory, memory)
        diagnostics.append(gates)
    return hidden, memory, diagnostics


def main() -> None:
    """Run an end-to-end masked sequence pass and verify a learning signal."""
    torch.manual_seed(31)
    cell = LSTMCell(input_size=3, hidden_size=5)
    tokens = torch.randn(2, 7, 3)
    lengths = torch.tensor([7, 4])
    hidden, memory, gates = run_sequence(cell, tokens, lengths)
    loss = hidden.square().mean() + 0.1 * memory.square().mean()
    loss.backward()
    forget_mean = gates[3].forget[1].mean().item()
    print(f"final state={tuple(hidden.shape)}, padded length protected, forget={forget_mean:.2f}")
    assert cell.gates.weight.grad is not None
    assert torch.isfinite(cell.gates.weight.grad).all()
    assert torch.isfinite(memory).all()
    print("ok: explicit gates preserve a masked, trainable cell-state path")


if __name__ == "__main__":
    main()
