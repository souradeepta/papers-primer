from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
out=Path(__file__).parent/'assets'/'dqn_replay_target.gif';out.parent.mkdir(exist_ok=True)
fig,ax=plt.subplots(figsize=(8,3));w=PillowWriter(fps=1)
with w.saving(fig,str(out),dpi=120):
 for title,text in [('act','epsilon-greedy action'),('store','transition enters replay buffer'),('sample','random minibatch'),('target','reward plus discounted max Q')]:
  ax.clear();ax.axis('off');ax.set_title(title,fontsize=16);ax.text(.5,.55,text,ha='center',va='center',transform=ax.transAxes,fontsize=20,bbox=dict(boxstyle='round,pad=.5',fc='#ffeadc',ec='#b45e2a'));w.grab_frame()
 for _ in range(3):w.grab_frame()
plt.close(fig);print(out,out.stat().st_size)
