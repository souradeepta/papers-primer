from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
out=Path(__file__).parent/'assets'/'residual_shortcut.gif'; out.parent.mkdir(exist_ok=True)
fig,ax=plt.subplots(figsize=(8,3)); writer=PillowWriter(fps=1)
frames=[('plain stack','x → layer → layer → y'),('residual branch','x → F(x)'),('shortcut','x ─────────────┐'),('addition','y = F(x) + x')]
with writer.saving(fig,str(out),dpi=120):
 for title,text in frames:
  ax.clear(); ax.axis('off'); ax.set_title(title,fontsize=16); ax.text(.5,.55,text,ha='center',va='center',transform=ax.transAxes,fontsize=22,bbox=dict(boxstyle='round,pad=.5',fc='#e4f2df',ec='#327a45')); writer.grab_frame()
 for _ in range(3): writer.grab_frame()
plt.close(fig); print(out, out.stat().st_size)
