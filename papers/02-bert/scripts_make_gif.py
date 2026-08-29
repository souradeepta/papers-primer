"""Generate assets/bidirectional_vs_causal.gif: an animation of a masked
query token's attention reach, contrasting BERT's fully bidirectional
self-attention against a GPT-style causal (left-only) self-attention over
the same sentence, one query position at a time.

This is the single most motion-worthy concept in the paper: the entire
argument for BERT over prior left-to-right or shallow-bidirectional
language models is that every layer lets a token see BOTH its left and
right context, not just what came before it (section 3, "BERT").

One-off generator; not part of the validated code/ smoke test."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

np.random.seed(0)
tokens = ["[CLS]", "the", "cat", "sat", "on", "[MASK]", "mat", "[SEP]"]
seq_len = len(tokens)
mask_pos = tokens.index("[MASK]")

# Toy affinity scores standing in for a trained model's Q.K^T for the
# masked position: strong pull toward both "sat" (left context) and "mat"
# (right context) -- the point being that "mat" is only reachable at all
# if the model is allowed to look right.
affinity = np.random.randn(seq_len) * 0.3
affinity[3] += 2.0   # sat  (left of [MASK])
affinity[6] += 2.2   # mat  (right of [MASK])
affinity[4] += 0.8   # on   (left of [MASK])

fig, (ax_bi, ax_causal) = plt.subplots(1, 2, figsize=(9, 4.2))
out_path = Path(__file__).parent / "assets" / "bidirectional_vs_causal.gif"
out_path.parent.mkdir(exist_ok=True)


def softmax(scores):
    e = np.exp(scores - scores.max())
    return e / e.sum()


# Bidirectional: [MASK] can attend to every position, left and right.
bi_scores = affinity.copy()
bi_weights = softmax(bi_scores)

# Causal: [MASK] can only attend to itself and positions to its left --
# "mat" (to the right) is architecturally invisible, its score set to
# -infinity before softmax, exactly as in a GPT-style decoder layer.
causal_scores = affinity.copy()
causal_scores[mask_pos + 1:] = -np.inf
causal_weights = softmax(causal_scores)

n_frames = 10
writer = PillowWriter(fps=2)
with writer.saving(fig, str(out_path), dpi=100):
    for frame in range(n_frames):
        # ramp attention in from 0 to its final value, frame by frame, to
        # give the GIF motion instead of a single static bar chart
        t = (frame + 1) / n_frames
        bi_frame = bi_weights * t
        causal_frame = causal_weights * t

        for ax, weights, title in (
            (ax_bi, bi_frame, "BERT: bidirectional\n([MASK] sees left AND right)"),
            (ax_causal, causal_frame, "GPT-style: causal\n([MASK] sees left only)"),
        ):
            ax.clear()
            colors = [
                "#DD8452" if i == mask_pos else
                ("#55A868" if weights[i] > 0.15 * t and i != mask_pos else "#4C72B0")
                for i in range(seq_len)
            ]
            ax.bar(range(seq_len), weights, color=colors)
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(tokens, rotation=45, fontsize=8)
            ax.set_ylim(0, 0.6)
            ax.set_ylabel("attention weight")
            ax.set_title(title, fontsize=9)
        fig.suptitle('Query = "[MASK]" predicting "sat ... on the ??? mat"', fontsize=10)
        fig.tight_layout()
        writer.grab_frame()
plt.close(fig)
print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")
