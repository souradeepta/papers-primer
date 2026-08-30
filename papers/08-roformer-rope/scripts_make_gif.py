"""Generate an explanatory RoPE rotation GIF, not a benchmark figure."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

out = Path(__file__).parent / "assets" / "relative_position_rotation.gif"
out.parent.mkdir(exist_ok=True)
q0, k0 = np.array([1.0, 0.0]), np.array([0.7, 0.7])
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
writer = PillowWriter(fps=1)
with writer.saving(fig, str(out), dpi=110):
    for p in range(9):
        for ax in axes: ax.clear(); ax.set(xlim=(-1.25,1.25), ylim=(-1.25,1.25), aspect="equal"); ax.axhline(0,color="gray",lw=.6); ax.axvline(0,color="gray",lw=.6)
        q = np.array([np.cos(.45*p), np.sin(.45*p)])
        k = np.array([np.cos(np.pi/4 + .45*(p-2)), np.sin(np.pi/4 + .45*(p-2))])
        axes[0].quiver(0,0,*q,angles="xy",scale_units="xy",scale=1,color="#2867ac",label=f"query at {p}")
        axes[0].quiver(0,0,*k,angles="xy",scale_units="xy",scale=1,color="#d84a3a",label=f"key at {p-2}")
        axes[0].set_title("same relative offset: +2"); axes[0].legend(loc="lower left",fontsize=8)
        k2 = np.array([np.cos(np.pi/4 + .45*(p-4)), np.sin(np.pi/4 + .45*(p-4))])
        axes[1].quiver(0,0,*q,angles="xy",scale_units="xy",scale=1,color="#2867ac",label=f"query at {p}")
        axes[1].quiver(0,0,*k2,angles="xy",scale_units="xy",scale=1,color="#d84a3a",label=f"key at {p-4}")
        axes[1].set_title("different relative offset: +4"); axes[1].legend(loc="lower left",fontsize=8)
        fig.suptitle("RoPE rotates both vectors; their angle tracks position difference")
        fig.tight_layout(); writer.grab_frame()
    for _ in range(3): writer.grab_frame()
plt.close(fig); print(f"wrote {out} ({out.stat().st_size} bytes)")
