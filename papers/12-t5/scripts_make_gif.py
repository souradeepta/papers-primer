"""Illustrate T5's text-to-text interface and span-corruption target."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
out=Path(__file__).parent/'assets'/'text_to_text_span_corruption.gif'; out.parent.mkdir(exist_ok=True)
frames=[('task prefix','summarize: long article → short summary'),('task prefix','translate English to German: hello → hallo'),('corrupt source','the <extra_id_0> sat on <extra_id_1> mat'),('decoder target','<extra_id_0> small cat <extra_id_1> the warm <extra_id_2>')]
fig,ax=plt.subplots(figsize=(9,3)); w=PillowWriter(fps=1)
with w.saving(fig,str(out),dpi=110):
 for label,text in frames:
  ax.clear(); ax.axis('off'); ax.set_title('T5: every task is text → text',fontsize=15)
  ax.text(.5,.62,label.upper(),transform=ax.transAxes,ha='center',fontsize=11,color='#555')
  ax.text(.5,.4,text,transform=ax.transAxes,ha='center',fontsize=14,bbox=dict(boxstyle='round,pad=.6',fc='#e3f1fb',ec='#2874a6'))
  w.grab_frame()
 for _ in range(3):w.grab_frame()
plt.close(fig);print(f'wrote {out} ({out.stat().st_size} bytes)')
