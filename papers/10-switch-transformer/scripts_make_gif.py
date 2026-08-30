"""Generate a synthetic, explanatory routing/balancing animation."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np
out=Path(__file__).parent/'assets'/'expert_load_balance.gif'; out.parent.mkdir(exist_ok=True)
loads=np.array([[12,1,1],[10,2,2],[8,3,3],[6,4,4],[5,5,4],[5,4,5]])
fig,(ax,ax2)=plt.subplots(1,2,figsize=(9,4)); writer=PillowWriter(fps=1)
with writer.saving(fig,str(out),dpi=110):
 for step,load in enumerate(loads):
  ax.clear(); ax2.clear(); colors=['#3b82b9','#e07a3f','#4f9b62']; ax.bar(['expert 0','expert 1','expert 2'],load,color=colors); ax.axhline(load.sum()/3,color='black',ls='--',label='ideal equal load'); ax.set(ylim=(0,14),ylabel='tokens routed',title=f'top-1 routing, balancing step {step}'); ax.legend(fontsize=8)
  for e,n in enumerate(load):
   ys=np.arange(n); ax2.scatter(np.full(n,e),ys,s=42,color=colors[e])
  ax2.set(xlim=(-.6,2.6),ylim=(-1,13),xticks=[0,1,2],xticklabels=['E0','E1','E2'],ylabel='token slot',title='tokens spread across expert queues')
  fig.suptitle('Auxiliary loss discourages a router collapse (synthetic illustration)'); fig.tight_layout(); writer.grab_frame()
 for _ in range(3): writer.grab_frame()
plt.close(fig); print(f'wrote {out} ({out.stat().st_size} bytes)')
