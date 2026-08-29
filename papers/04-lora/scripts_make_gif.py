"""Generate assets/lora_rank_scaling.gif: an animation of LoRA's single
most-cited practical property -- trainable parameter count grows only
*linearly* with rank r, while it stays orders of magnitude below the full
fine-tuning parameter count even as r increases.

Setup mirrors the paper's own numbers as closely as an illustrative,
scaled figure can: GPT-3 175B's hidden dimension is d_model=12288 (a fact
reported in the GPT-3 paper, Brown et al. 2020, not this paper), so a
single Wq or Wv attention projection matrix is 12288 x 12288 =
150,994,944 full parameters. LoRA replaces training that whole matrix
with two skinny matrices A (r x 12288) and B (12288 x r), i.e.
r * (12288 + 12288) trainable parameters. This script animates that
"LoRA params" bar growing as r doubles (r = 1, 2, 4, 8, 16, 32, 64), next
to a fixed reference line for the full matrix's parameter count -- the
same qualitative shape as the paper's headline claim that GPT-3
fine-tuning needs ~10,000x fewer trainable parameters with LoRA (paper,
abstract and section 4.2, for the actual end-to-end multi-layer,
multi-matrix number; this figure isolates the single-matrix trend that
drives that headline number).

One-off generator; not part of the validated code/ smoke test.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

d_model = 12288  # GPT-3 175B hidden dimension (Brown et al. 2020), for scale
full_params = d_model * d_model  # one Wq (or Wv) attention projection matrix

r_values = [1, 2, 4, 8, 16, 32, 64]
lora_params = [r * (d_model + d_model) for r in r_values]

out_path = Path(__file__).parent / "assets" / "lora_rank_scaling.gif"
out_path.parent.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 4.8))
writer = PillowWriter(fps=1.2)

with writer.saving(fig, str(out_path), dpi=100):
    for frame in range(len(r_values)):
        ax.clear()
        upto = frame + 1
        xs = r_values[:upto]
        ys = lora_params[:upto]

        ax.bar(
            [str(r) for r in xs], ys, color="#4C72B0", label="LoRA trainable params: r x (d+k)",
        )
        ax.axhline(
            full_params, color="#C44E52", linewidth=2, linestyle="--",
            label=f"Full fine-tune: one {d_model}x{d_model} matrix = {full_params:,}",
        )
        ax.set_yscale("log")
        ax.set_ylim(1e3, 1e9)
        ax.set_xlabel("LoRA rank r")
        ax.set_ylabel("trainable parameters (log scale)")
        ax.set_title(
            "LoRA trainable params grow linearly with rank r,\n"
            "but stay orders of magnitude below full fine-tuning\n"
            "(single attention projection matrix, GPT-3 175B scale)",
            fontsize=9.5,
        )
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3, axis="y")

        reduction = full_params / ys[-1]
        ax.annotate(
            f"r={xs[-1]}: {ys[-1]:,} params\n({reduction:,.0f}x fewer than full)",
            xy=(len(xs) - 1, ys[-1]),
            xytext=(0, 12),
            textcoords="offset points",
            fontsize=8,
            ha="center",
        )
        fig.tight_layout()
        writer.grab_frame()
    # Hold the final frame a bit longer so the completed comparison is visible.
    for _ in range(3):
        writer.grab_frame()
plt.close(fig)
print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")
