"""Generate a GIF showing why online-softmax attention needs a running-max
correction, using a single query row against streamed blocks of keys.

This is an illustrative simulation of the running statistics in
FlashAttention's Algorithm 1 (Dao et al., 2022) for ONE query row: as each
new block of keys/values streams in, the block's own local max can exceed
the running max seen so far, so the running max m_i is updated, and the
already-accumulated normalizer l_i must be rescaled by exp(m_old - m_new)
to stay a valid, consistent softmax normalizer. The right panel contrasts
this against a "naive" approach that computes each block's softmax
contribution against only that block's own local max and just sums the
blocks -- i.e. it skips the running-max correction entirely. Because later
blocks are never rescaled relative to earlier ones (or vice versa), the
naive total systematically overstates the true normalizer. This is a
teaching simulation with synthetic scores, not a reproduction of the
paper's measured figures.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

rng = np.random.default_rng(3)
num_keys = 24
block_size = 4
num_blocks = num_keys // block_size

# Synthetic (already-scaled) query-key dot-product scores for one query row.
# The last block is made to contain the true row max, so the running max
# has to jump upward late in the stream -- the scenario online softmax must
# handle correctly.
scores = rng.normal(loc=0.0, scale=1.0, size=num_keys)
scores[-block_size:] += 4.0

block_colors = plt.cm.viridis(np.linspace(0.15, 0.9, num_blocks))

running_max = []
running_l_correct = []
running_l_naive = []

m_i = -np.inf
l_i = 0.0
l_naive = 0.0
for b in range(num_blocks):
    block = scores[b * block_size : (b + 1) * block_size]
    m_block = block.max()
    m_new = max(m_i, m_block)

    # Correct online-softmax update: rescale the old accumulator by
    # exp(m_i - m_new) so it stays referenced to the new shared max.
    alpha = np.exp(m_i - m_new) if np.isfinite(m_i) else 0.0
    l_i = alpha * l_i + np.exp(block - m_new).sum()

    # Naive (buggy) alternative: normalize each block only against its OWN
    # local max and sum the blocks directly, with no cross-block rescaling.
    l_naive += np.exp(block - m_block).sum()

    m_i = m_new
    running_max.append(m_i)
    running_l_correct.append(l_i)
    running_l_naive.append(l_naive)

exact_l = np.exp(scores - m_i).sum()  # ground-truth normalizer for the full row

out_path = Path(__file__).parent / "assets" / "online_softmax_correction.gif"
out_path.parent.mkdir(exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
writer = PillowWriter(fps=1)
with writer.saving(fig, str(out_path), dpi=100):
    for b in range(num_blocks):
        ax1.clear()
        ax2.clear()

        seen = (b + 1) * block_size
        colors = [block_colors[j // block_size] for j in range(seen)]
        ax1.bar(range(seen), scores[:seen], color=colors)
        ax1.axhline(
            running_max[b], color="crimson", linestyle="--",
            label=f"running max m_i = {running_max[b]:.2f}",
        )
        ax1.set_xlim(-1, num_keys)
        ax1.set_ylim(scores.min() - 1, scores.max() + 1)
        ax1.set_title(f"streaming key/value block {b + 1}/{num_blocks}")
        ax1.set_xlabel("key index")
        ax1.set_ylabel("query . key score")
        ax1.legend(loc="upper left", fontsize=8)

        steps = np.arange(1, b + 2)
        ax2.plot(steps, running_l_correct[: b + 1], "o-", color="#2E7D32",
                  label="correct l_i (running-max rescaled)")
        ax2.plot(steps, running_l_naive[: b + 1], "o--", color="#C62828",
                  label="naive l_i (no rescale, local max only)")
        ax2.axhline(exact_l, color="gray", linestyle=":",
                     label=f"exact normalizer = {exact_l:.2f}")
        ax2.set_xlim(0.5, num_blocks + 0.5)
        ax2.set_ylim(0, max(running_l_naive) * 1.15)
        ax2.set_xlabel("blocks streamed so far")
        ax2.set_ylabel("softmax normalizer estimate")
        ax2.set_title("running-max correction keeps l_i exact")
        ax2.legend(loc="upper left", fontsize=8)

        fig.tight_layout()
        writer.grab_frame()
    for _ in range(3):
        writer.grab_frame()
plt.close(fig)
print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")
