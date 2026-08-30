"""Create an illustrative, not paper-result, skip-gram training GIF."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

out = Path(__file__).parent / "assets" / "skipgram_pairs.gif"
out.parent.mkdir(exist_ok=True)
frames = [("Sentence", ["coffee", "with", "tea"], "slide a context window"),
          ("Center word", ["coffee", "with", "tea"], "center = with"),
          ("Positive pairs", ["coffee ↔ with", "with ↔ tea"], "raise their scores"),
          ("Sampled noise", ["with ↔ tractor"], "lower its score")]
fig, ax = plt.subplots(figsize=(8, 3.2))
writer = PillowWriter(fps=1)
with writer.saving(fig, str(out), dpi=120):
    for title, items, caption in frames:
        ax.clear(); ax.axis("off"); ax.set_title(title, fontsize=16)
        for i, item in enumerate(items):
            ax.text(.16 + .31 * i, .58, item, transform=ax.transAxes, ha="center", va="center",
                    fontsize=17, bbox=dict(boxstyle="round,pad=.45", fc="#dceef8", ec="#2b6f9c"))
        ax.text(.5, .18, caption, transform=ax.transAxes, ha="center", fontsize=13)
        writer.grab_frame()
    for _ in range(3): writer.grab_frame()
plt.close(fig)
print(f"wrote {out} ({out.stat().st_size} bytes)")
