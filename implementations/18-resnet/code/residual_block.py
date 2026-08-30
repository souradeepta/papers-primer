"""Show the defining invariant of an identity residual block without NumPy."""

# Reading guide: follow the named helpers in data-flow order, then inspect the
# assertions at the bottom. Change one toy input at a time and rerun the file.
from __future__ import annotations

def residual_block(x: list[float], residual: list[float]) -> list[float]:
    return [a + b for a, b in zip(x, residual)]

def main() -> None:
    x = [1.0, -2.0, 3.0]
    identity = residual_block(x, [0.0, 0.0, 0.0])
    refined = residual_block(x, [.2, -.1, .4])
    print(f"identity path: {identity}; learned residual path: {refined}")
    assert identity == x and refined != x
    print("ok: zero residual preserves the input through the shortcut")
if __name__ == '__main__': main()

