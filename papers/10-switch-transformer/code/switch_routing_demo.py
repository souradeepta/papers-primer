"""Top-1 Switch routing with capacity and the paper's load-balancing loss."""
from __future__ import annotations
import torch

def route(logits: torch.Tensor, capacity: int):
    probs = logits.softmax(-1)
    expert = probs.argmax(-1)
    accepted = torch.zeros_like(expert, dtype=torch.bool)
    counts = torch.zeros(logits.shape[1], dtype=torch.long)
    for token, e in enumerate(expert.tolist()):
        if counts[e] < capacity:
            accepted[token] = True; counts[e] += 1
    # Switch auxiliary loss: N * sum_i(f_i * P_i), with f hard routing fraction
    fraction = torch.stack([(expert == i).float().mean() for i in range(logits.shape[1])])
    mean_probability = probs.mean(0)
    aux = logits.shape[1] * (fraction * mean_probability).sum()
    return expert, accepted, counts, aux

def main() -> None:
    # Deliberately collapse 7 tokens onto expert 0; capacity admits only 2.
    logits = torch.tensor([[5., 0., 0.]] * 7 + [[0., 5., 0.], [0., 0., 5.]])
    expert, accepted, counts, collapsed = route(logits, capacity=2)
    uniform_logits = torch.zeros(9, 3)
    _, _, _, uniform = route(uniform_logits, capacity=9)
    print(f"top-1 assignments: {expert.tolist()}")
    print(f"accepted per expert: {counts.tolist()}; dropped: {(~accepted).sum().item()}")
    print(f"aux loss collapsed={collapsed:.3f}; uniform-probability={uniform:.3f}")
    assert counts.tolist() == [2, 1, 1] and (~accepted).sum().item() == 5
    assert collapsed > uniform
    print("ok: capacity drops overflow and the balancing loss penalizes collapsed routing")
if __name__ == '__main__': main()
