"""Visualize the toy DPO objective's preferred and rejected relative log-ratios."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import torch
out=Path(__file__).parent/'assets'/'preference_margin.gif'; out.parent.mkdir(exist_ok=True)
ref=torch.tensor([.2,-.1,.4]); p=torch.nn.Parameter(torch.zeros(3)); opt=torch.optim.SGD([p],lr=.18); chosen=[]; rejected=[]
for _ in range(35):
    lp=p.log_softmax(0); lr=ref.log_softmax(0); c=lp[0]-lr[0]; r=lp[1]-lr[1]
    opt.zero_grad(); (-torch.nn.functional.logsigmoid(.5*(c-r))).backward(); opt.step(); chosen.append(c.item()); rejected.append(r.item())
fig,ax=plt.subplots(figsize=(7,4)); writer=PillowWriter(fps=4)
with writer.saving(fig,str(out),dpi=110):
 for i in range(2,len(chosen)+1):
  ax.clear(); ax.plot(chosen[:i],label='chosen log π/log πref',color='#2474b5',lw=2); ax.plot(rejected[:i],label='rejected log π/log πref',color='#d4533c',lw=2); ax.fill_between(range(i),rejected[:i],chosen[:i],color='#78b876',alpha=.25,label='implicit reward margin'); ax.set(xlabel='gradient step',ylabel='relative log-probability',title='DPO widens the preferred response margin'); ax.legend(fontsize=8); fig.tight_layout(); writer.grab_frame()
for _ in range(4): writer.grab_frame()
plt.close(fig); print(f'wrote {out} ({out.stat().st_size} bytes)')
