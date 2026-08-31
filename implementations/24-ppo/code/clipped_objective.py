"""PPO clipped policy surrogate with value and entropy losses."""
from __future__ import annotations
import torch


class ActorCritic(torch.nn.Module):
    """Small categorical policy and scalar value function sharing an encoder."""
    def __init__(self) -> None:
        super().__init__(); self.body=torch.nn.Sequential(torch.nn.Linear(4,16),torch.nn.Tanh())
        self.policy,self.value=torch.nn.Linear(16,2),torch.nn.Linear(16,1)
    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        h=self.body(states); return self.policy(h),self.value(h).squeeze(1)


def main() -> None:
    torch.manual_seed(24); model=ActorCritic(); states=torch.randn(12,4); actions=torch.randint(0,2,(12,))
    with torch.no_grad():
        old_logits, old_values=model(states); old_logp=torch.distributions.Categorical(logits=old_logits).log_prob(actions)
        returns=old_values+torch.randn(12); advantages=returns-old_values
    logits, values=model(states); distribution=torch.distributions.Categorical(logits=logits)
    ratio=(distribution.log_prob(actions)-old_logp).exp(); clipped=ratio.clamp(.8,1.2)
    policy_loss=-torch.minimum(ratio*advantages,clipped*advantages).mean()
    value_loss=torch.nn.functional.mse_loss(values,returns); entropy=distribution.entropy().mean()
    total=policy_loss+.5*value_loss-.01*entropy
    total.backward()
    print(f"policy={policy_loss.item():.3f}, value={value_loss.item():.3f}, entropy={entropy.item():.3f}")
    assert model.policy.weight.grad is not None and torch.isfinite(total)
    print("ok: PPO combines clipped policy improvement, value fitting, and exploration")


if __name__ == "__main__":
    main()
