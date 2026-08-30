"""Illustrate reversible SentencePiece-style whitespace and best-path pieces."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
out=Path(__file__).parent/'assets'/'reversible_segmentation.gif'; out.parent.mkdir(exist_ok=True)
frames=[('raw text','hello world',['hello world']),('visible whitespace','▁hello▁world',['▁','hello','▁','world']),('candidate pieces','▁hello | ▁world',['▁hello','▁world']),('decode','hello world',['hello world'])]
fig,ax=plt.subplots(figsize=(8,3)); writer=PillowWriter(fps=1)
with writer.saving(fig,str(out),dpi=120):
 for title,text,pieces in frames:
  ax.clear(); ax.axis('off'); ax.set_title(title,fontsize=15); x=.06
  for p in pieces:
   width=.07*len(p)+.12; ax.text(x,.55,p,transform=ax.transAxes,ha='left',va='center',fontsize=18,bbox=dict(boxstyle='round,pad=.45',fc='#dceef8',ec='#2b6f9c')); x+=width
  ax.text(.5,.18,f'concatenate pieces → {text}',transform=ax.transAxes,ha='center',fontsize=12); writer.grab_frame()
 for _ in range(3): writer.grab_frame()
plt.close(fig); print(f'wrote {out} ({out.stat().st_size} bytes)')
