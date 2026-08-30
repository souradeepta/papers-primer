from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
out=Path(__file__).parent/'assets'/'adversarial_game.gif'; out.parent.mkdir(exist_ok=True)
fig,ax=plt.subplots(figsize=(8,3)); writer=PillowWriter(fps=1)
for_text=[('real data','real samples → discriminator'),('noise','z → generator → fake sample'),('D update','separate real from fake'),('G update','make fake score look real')]
with writer.saving(fig,str(out),dpi=120):
 for title,text in for_text:
  ax.clear(); ax.axis('off'); ax.set_title(title,fontsize=16); ax.text(.5,.55,text,ha='center',va='center',transform=ax.transAxes,fontsize=21,bbox=dict(boxstyle='round,pad=.5',fc='#fde4df',ec='#a34232')); writer.grab_frame()
 for _ in range(3): writer.grab_frame()
plt.close(fig); print(out, out.stat().st_size)
