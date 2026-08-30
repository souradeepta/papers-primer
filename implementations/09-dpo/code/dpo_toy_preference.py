"""A compact Direct Preference Optimization training loop.

DPO trains a policy from prompt, chosen-response, rejected-response triples
without fitting a separate reward model. This script uses a tiny
prompt-conditioned categorical language model and the paper's log-ratio
objective. It keeps the frozen reference policy, batched preferences, beta
temperature, and gradient updates used in real DPO training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional


class TinyPolicy(torch.nn.Module):
    """A one-token language model whose embedding row is response logits."""

    def __init__(self, prompts: int, responses: int) -> None:
        super().__init__()
        self.logits = torch.nn.Embedding(prompts, responses)

    def log_probability(self, prompts: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        """Gather log pi(response given prompt) for every pair in a batch."""
        log_distribution = functional.log_softmax(self.logits(prompts), dim=-1)
        return log_distribution.gather(1, responses[:, None]).squeeze(1)


def dpo_loss(
    policy: TinyPolicy, reference: TinyPolicy, prompts: torch.Tensor,
    chosen: torch.Tensor, rejected: torch.Tensor, beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Implement the DPO chosen/rejected log-probability-ratio objective."""
    policy_margin = policy.log_probability(prompts, chosen) - policy.log_probability(prompts, rejected)
    # The reference is fixed: it defines how far the trained policy has moved.
    with torch.no_grad():
        reference_margin = reference.log_probability(prompts, chosen) - reference.log_probability(prompts, rejected)
    advantage = beta * (policy_margin - reference_margin)
    return -functional.logsigmoid(advantage).mean(), advantage.mean()


def main() -> None:
    torch.manual_seed(9)
    prompts = torch.tensor([0, 0, 1, 1, 2, 2])
    chosen = torch.tensor([0, 2, 1, 3, 2, 0])
    rejected = torch.tensor([1, 3, 0, 2, 3, 1])
    reference = TinyPolicy(prompts=3, responses=4)
    policy = TinyPolicy(prompts=3, responses=4)
    policy.load_state_dict(reference.state_dict())
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.15)

    before_loss, before_advantage = dpo_loss(policy, reference, prompts, chosen, rejected, beta=0.5)
    for _ in range(100):
        optimizer.zero_grad()
        loss, _ = dpo_loss(policy, reference, prompts, chosen, rejected, beta=0.5)
        loss.backward()
        optimizer.step()
    after_loss, after_advantage = dpo_loss(policy, reference, prompts, chosen, rejected, beta=0.5)

    print(f"DPO loss: {before_loss.item():.4f} -> {after_loss.item():.4f}")
    print(f"mean reference-relative preference advantage: {before_advantage.item():.4f} -> {after_advantage.item():.4f}")
    assert after_loss < before_loss * 0.35
    assert after_advantage > before_advantage + 2.0
    print("ok: DPO increases the chosen response's likelihood relative to both rejected and reference policies")


if __name__ == "__main__":
    main()
