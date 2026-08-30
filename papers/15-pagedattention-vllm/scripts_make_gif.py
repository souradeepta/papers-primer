from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
out=Path(__file__).parent/'assets'/'paged_kv_blocks.gif';out.parent.mkdir(exist_ok=True)
frames=[('request A logical blocks',[2,7,4]),('request B shares prefix',[2,9]),('A finishes: block 2 retained',[2]),('B finishes: block 2 reclaimed',[])]
fig,ax=plt.subplots(figsize=(8,3));w=PillowWriter(fps=1)
with w.saving(fig,str(out),dpi=110):
 for title,blocks in frames:
  ax.clear();ax.set(xlim=(0,10),ylim=(0,2),yticks=[]);ax.set_title(title)
  for i,b in enumerate(blocks):ax.add_patch(plt.Rectangle((i*2+.5,.7),1.5,.6,color='#68a9d4'));ax.text(i*2+1.25,1,f'physical {b}',ha='center')
  w.grab_frame()
 for _ in range(3):w.grab_frame()
plt.close(fig);print(out,out.stat().st_size)
