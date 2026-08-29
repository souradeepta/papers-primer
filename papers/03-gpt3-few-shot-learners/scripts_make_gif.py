"""Generate assets/incontext_learning_curve.gif: an animation of the paper's
single most-cited empirical trend -- aggregate task performance rising as
more demonstrations (K) are packed into the prompt, with the gain being
*larger* for a bigger model, exactly the qualitative shape of Figure 1.2 in
Brown et al. 2020 ("Aggregate few-shot performance... as a function of the
number of in-context examples").

The curve values below are illustrative synthetic numbers with the same
qualitative shape as Figure 1.2 (larger model = higher, steeper improvement
with more examples) -- they are NOT the paper's actual benchmark numbers,
which the paper reports as an aggregate over 42 accuracy-denominated tasks
and does not reduce to a single closed-form curve.

One-off generator; not part of the validated code/ smoke test."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

np.random.seed(0)

k_values = [0, 1, 2, 4, 8, 16, 32]  # number of in-context (few-shot) examples

# Illustrative synthetic accuracy curves -- same qualitative shape as the
# paper's Figure 1.2: both models improve as K grows, but the larger model
# starts higher (better zero-shot) AND improves faster (bigger few-shot
# gain), which is the paper's central scaling claim about in-context
# learning specifically (not just about zero-shot performance).
small_model_final = np.array([28, 31, 33, 35, 37, 38, 39])
large_model_final = np.array([42, 51, 58, 64, 69, 72, 74])

out_path = Path(__file__).parent / "assets" / "incontext_learning_curve.gif"
out_path.parent.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
n_frames = len(k_values)
writer = PillowWriter(fps=1.2)

with writer.saving(fig, str(out_path), dpi=100):
    for frame in range(n_frames):
        ax.clear()
        upto = frame + 1
        x = k_values[:upto]
        ax.plot(
            x, small_model_final[:upto], marker="o", color="#4C72B0",
            label="smaller model (e.g. 1.3B)", linewidth=2,
        )
        ax.plot(
            x, large_model_final[:upto], marker="o", color="#DD8452",
            label="larger model (e.g. 175B)", linewidth=2,
        )
        ax.set_xscale("symlog", base=2)
        ax.set_xticks(k_values)
        ax.set_xticklabels(k_values)
        ax.set_xlim(-0.5, 34)
        ax.set_ylim(20, 80)
        ax.set_xlabel("number of in-context (few-shot) examples, K")
        ax.set_ylabel("illustrative aggregate accuracy (%)")
        ax.set_title(
            "In-context learning: more examples in the prompt help,\n"
            "and help MORE for a larger model (cf. paper Fig. 1.2)",
            fontsize=10,
        )
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
        if upto <= len(k_values):
            ax.annotate(
                f"K={x[-1]}", xy=(x[-1], large_model_final[upto - 1]),
                xytext=(5, 8), textcoords="offset points", fontsize=9,
            )
        fig.tight_layout()
        writer.grab_frame()
    # Hold the final frame a bit longer so the completed curves are visible.
    for _ in range(3):
        writer.grab_frame()
plt.close(fig)
print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")
