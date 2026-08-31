"""DQN replay batch, target network, and Bellman regression update."""
from __future__ import annotations
import torch


class QNetwork(torch.nn.Module):
    """Map toy states to values of two discrete actions."""
    def __init__(self) -> None:
        super().__init__(); self.network=torch.nn.Sequential(torch.nn.Linear(4,16),torch.nn.ReLU(),torch.nn.Linear(16,2))
    def forward(self, states: torch.Tensor) -> torch.Tensor: return self.network(states)


def main() -> None:
    torch.manual_seed(26); online,target=QNetwork(),QNetwork(); target.load_state_dict(online.state_dict())
    optimizer=torch.optim.Adam(online.parameters(),lr=.03)
    states,next_states=torch.randn(16,4),torch.randn(16,4); actions=torch.randint(0,2,(16,))
    rewards,done=torch.randn(16),torch.rand(16)<.2
    with torch.no_grad():
        bellman=rewards+.99*(~done)*target(next_states).max(1).values
    prediction=online(states).gather(1,actions[:,None]).squeeze(1)
    loss=torch.nn.functional.smooth_l1_loss(prediction,bellman)
    optimizer.zero_grad();loss.backward();optimizer.step()
    print(f"replay Bellman loss: {loss.item():.3f}; terminal transitions: {done.sum().item()}")
    assert online.network[0].weight.grad is not None and torch.isfinite(loss)
    print("ok: DQN regresses replayed action values to frozen target-network returns")


if __name__ == "__main__":
    main()
