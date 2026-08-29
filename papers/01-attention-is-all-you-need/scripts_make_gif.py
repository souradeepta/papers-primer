"""Generate assets/attention_weights.gif: a toy attention-weight bar chart
animating as the query position sweeps across a short sequence, showing
which tokens each query attends to most (the core dynamic in section 3.2).
One-off generator; not part of the validated code/ smoke test."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

np.random.seed(0)
seq_len = 6
tokens = ["The", "cat", "sat", "on", "the", "mat"]
# A toy affinity matrix standing in for learned Q.K^T scores: "cat" and
# "sat" favor each other, "the" tokens favor their following noun.
base = np.random.randn(seq_len, seq_len) * 0.3
base[1, 2] += 2.0  # cat -> sat
base[2, 1] += 1.5  # sat -> cat
base[0, 1] += 1.2  # The -> cat
base[4, 5] += 1.8  # the -> mat

fig, ax = plt.subplots(figsize=(4, 4))
out_path = Path(__file__).parent / "assets" / "attention_weights.gif"
out_path.parent.mkdir(exist_ok=True)

writer = PillowWriter(fps=1)
with writer.saving(fig, str(out_path), dpi=100):
    for query_pos in range(seq_len):
        ax.clear()
        row = base[query_pos]
        weights = np.exp(row) / np.exp(row).sum()
        colors = ["#4C72B0" if i != query_pos else "#DD8452" for i in range(seq_len)]
        ax.bar(range(seq_len), weights, color=colors)
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(tokens, rotation=45)
        ax.set_ylim(0, 1)
        ax.set_ylabel("attention weight")
        ax.set_title(f'Query = "{tokens[query_pos]}" — softmax(QK^T/sqrt(d_k))')
        writer.grab_frame()
plt.close(fig)
print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")
