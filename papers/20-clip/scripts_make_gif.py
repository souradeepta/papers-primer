from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

out=Path(__file__).parent/'assets'/'contrastive_pairs.gif'; out.parent.mkdir(exist_ok=True)
fig,ax=plt.subplots(figsize=(8,3)); writer=PillowWriter(fps=1)
frames=[('paired data','image + caption'),('two encoders','image encoder | text encoder'),('similarity matrix','every image compared to every text'),('contrastive target','raise diagonal, lower mismatches')]
with writer.saving(fig,str(out),dpi=120):
 for title,text in frames:
  ax.clear(); ax.axis('off'); ax.set_title(title,fontsize=16)
  ax.text(.5,.55,text,ha='center',va='center',transform=ax.transAxes,fontsize=20,bbox=dict(boxstyle='round,pad=.5',fc='#e8e0f7',ec='#6645a3'))
  writer.grab_frame()
 for _ in range(3): writer.grab_frame()
plt.close(fig); print(out, out.stat().st_size)
