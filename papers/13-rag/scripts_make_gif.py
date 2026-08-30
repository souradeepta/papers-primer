from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
out=Path(__file__).parent/'assets'/'retrieval_marginalization.gif';out.parent.mkdir(exist_ok=True)
frames=[('query','Which planet is red?'),('top-k retrieval','Mars passage  0.73\nJupiter passage  0.27'),('generator evidence','P(answer= Mars | passage)'),('marginal answer','0.73 × evidence₁ + 0.27 × evidence₂')]
fig,ax=plt.subplots(figsize=(8,3));w=PillowWriter(fps=1)
with w.saving(fig,str(out),dpi=110):
 for title,body in frames:
  ax.clear();ax.axis('off');ax.set_title('RAG combines retrieved memory with a generator',fontsize=14);ax.text(.5,.48,title+'\n\n'+body,ha='center',va='center',transform=ax.transAxes,fontsize=14,bbox=dict(boxstyle='round,pad=.7',fc='#e5f2fa'));w.grab_frame()
 for _ in range(3):w.grab_frame()
plt.close(fig);print(out,out.stat().st_size)
