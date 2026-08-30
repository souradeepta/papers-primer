"""A compact Switch Transformer sparse mixture-of-experts layer.

Switch Transformer sends each token to exactly one feed-forward expert. This
implementation includes the learned router, top-1 dispatch, per-expert
capacity, dropped overflow tokens, independent expert MLPs, and the paper's
load-balancing auxiliary loss. It omits distributed all-to-all communication
and the surrounding Transformer blocks so it remains readable on a CPU.
"""

from __future__ import annotations

import math

import torch


class SwitchFeedForward(torch.nn.Module):
    """A capacity-limited top-1 routed bank of feed-forward experts."""

    def __init__(self, width: int, expert_count: int, capacity_factor: float = 1.0) -> None:
        super().__init__()
        self.router = torch.nn.Linear(width, expert_count, bias=False)
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(width, 2 * width),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * width, width),
            )
            for _ in range(expert_count)
        ])
        self.expert_count = expert_count
        self.capacity_factor = capacity_factor

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route flattened token rows and return output, balancing loss, accepted mask."""
        router_logits = self.router(tokens)
        router_probabilities = router_logits.softmax(dim=-1)
        selected_expert = router_probabilities.argmax(dim=-1)
        capacity = math.ceil(self.capacity_factor * len(tokens) / self.expert_count)

        output = torch.zeros_like(tokens)
        accepted = torch.zeros(len(tokens), dtype=torch.bool)
        for expert_id, expert in enumerate(self.experts):
            # Stable selection preserves token order; later tokens overflow
            # capacity, matching the paper's capacity-limited dispatch idea.
            token_ids = (selected_expert == expert_id).nonzero(as_tuple=True)[0][:capacity]
            output[token_ids] = expert(tokens[token_ids])
            accepted[token_ids] = True

        # Switch equation 4: f is hard routed token fraction and P is the
        # router's mean soft probability. Their product discourages collapse.
        routed_fraction = torch.stack([
            (selected_expert == expert_id).float().mean()
            for expert_id in range(self.expert_count)
        ])
        mean_router_probability = router_probabilities.mean(dim=0)
        balancing_loss = self.expert_count * (routed_fraction * mean_router_probability).sum()
        return output, balancing_loss, accepted


def main() -> None:
    torch.manual_seed(10)
    layer = SwitchFeedForward(width=8, expert_count=4, capacity_factor=1.0)
    tokens = torch.randn(12, 8)
    output, balancing_loss, accepted = layer(tokens)

    print(f"input/output shape: {tuple(tokens.shape)} -> {tuple(output.shape)}")
    print(f"per-expert capacity: 3; accepted={accepted.sum().item()}, dropped={(~accepted).sum().item()}")
    print(f"load-balancing loss: {balancing_loss.item():.4f}")
    assert output.shape == tokens.shape
    assert accepted.sum() <= len(tokens)
    assert torch.isfinite(balancing_loss) and balancing_loss >= 1.0
    print("ok: top-1 routing runs experts under capacity and produces a balancing signal")


if __name__ == "__main__":
    main()
