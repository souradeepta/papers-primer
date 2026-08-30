"""Generate an illustrative fixed-compute loss-valley GIF for this primer.

The curve visualizes the paper's question, not its measured loss values:
given a fixed budget C ~= 6ND, choosing more parameters N forces fewer
tokens D.  The plotted loss uses a simple symmetric teaching function, so
the minimum is visibly near the balanced allocation.  It is deliberately
labelled illustrative rather than presenting synthetic data as Chinchilla
measurements.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter


def loss(n: np.ndarray, compute: float) -> np.ndarray:
    d = compute / n
    return 1 + 100 / np.sqrt(n) + 100 / np.sqrt(d)


out_path = Path(__file__).parent / "assets" / "fixed_compute_valley.gif"
out_path.parent.mkdir(exist_ok=True)
compute = 1_000_000.0
parameters = np.logspace(1, 5, 300)
losses = loss(parameters, compute)
best_index = int(np.argmin(losses))

fig, ax = plt.subplots(figsize=(7, 4.5))
writer = PillowWriter(fps=1.2)
with writer.saving(fig, str(out_path), dpi=100):
    for endpoint in np.linspace(40, len(parameters), 7, dtype=int):
        ax.clear()
        ax.plot(parameters[:endpoint], losses[:endpoint], color="#4C72B0", linewidth=2)
        ax.set_xscale("log")
        ax.set_xlabel("model parameters N (normalized, log scale)")
        ax.set_ylabel("illustrative final loss")
        ax.set_title("At fixed compute, both too-small and too-large models lose")
        ax.grid(alpha=0.25)
        if endpoint > best_index:
            n = parameters[best_index]
            ax.scatter([n], [losses[best_index]], color="#C44E52", zorder=3)
            ax.annotate(
                "balanced allocation\nN = D = 1,000",
                xy=(n, losses[best_index]), xytext=(1_800, losses[best_index] + 0.35),
                arrowprops={"arrowstyle": "->", "color": "#333"}, fontsize=9,
            )
        ax.text(0.02, 0.04, "Illustrative loss curve; not paper measurements", transform=ax.transAxes,
                fontsize=8, color="#555")
        fig.tight_layout()
        writer.grab_frame()
    for _ in range(2):
        writer.grab_frame()
plt.close(fig)
print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")
