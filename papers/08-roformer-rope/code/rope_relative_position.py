"""A CPU-only demonstration of the relative-position identity behind RoPE."""
from __future__ import annotations

import torch


def rope(x: torch.Tensor, position: int, base: float = 10_000.0) -> torch.Tensor:
    """Rotate consecutive pairs of x by position-dependent RoPE angles."""
    assert x.ndim == 1 and x.numel() % 2 == 0
    pair_index = torch.arange(0, x.numel(), 2, dtype=x.dtype)
    theta = base ** (-pair_index / x.numel())
    angle = position * theta
    even, odd = x[0::2], x[1::2]
    rotated = torch.empty_like(x)
    rotated[0::2] = even * torch.cos(angle) - odd * torch.sin(angle)
    rotated[1::2] = even * torch.sin(angle) + odd * torch.cos(angle)
    return rotated


def score(q: torch.Tensor, k: torch.Tensor, q_pos: int, k_pos: int) -> float:
    return torch.dot(rope(q, q_pos), rope(k, k_pos)).item()


def main() -> None:
    torch.manual_seed(8)
    q, k = torch.randn(8), torch.randn(8)
    # All three pairs have q_pos - k_pos == 3; their scores must be equal.
    same_offset = [score(q, k, *pair) for pair in [(3, 0), (8, 5), (21, 18)]]
    other_offset = score(q, k, 4, 0)
    print("offset +3 scores:", [f"{value:.7f}" for value in same_offset])
    print(f"offset +4 score:  {other_offset:.7f}")
    assert torch.allclose(torch.tensor(same_offset), torch.full((3,), same_offset[0]), atol=1e-6)
    assert abs(same_offset[0] - other_offset) > 1e-4
    for pos in (0, 1, 7):
        assert torch.allclose(rope(q, pos).norm(), q.norm(), atol=1e-6)
    print("ok: equal offsets give equal RoPE scores; rotations preserve norms")


if __name__ == "__main__":
    main()
