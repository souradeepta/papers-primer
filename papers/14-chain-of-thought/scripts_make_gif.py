from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
out=Path(__file__).parent/'assets'/'reasoning_traces.gif';out.parent.mkdir(exist_ok=True)
frames=[('answer-only','Q: 3 + 4?\nA: 7'),('chain of thought','Q: 3 + 4?\nA: Start at 3; add 4; therefore 7.'),('later self-consistency','trace answers: 7, 8, 7, 7, 7\nmajority: 7')]
fig,ax=plt.subplots(figsize=(8,3));w=PillowWriter(fps=1)
with w.saving(fig,str(out),dpi=110):
 for title,text in frames:
  ax.clear();ax.axis('off');ax.set_title(title,fontsize=15);ax.text(.5,.45,text,transform=ax.transAxes,ha='center',va='center',fontsize=14,bbox=dict(boxstyle='round,pad=.7',fc='#e6f2fb'));w.grab_frame()
 for _ in range(3):w.grab_frame()
plt.close(fig);print(out,out.stat().st_size)
