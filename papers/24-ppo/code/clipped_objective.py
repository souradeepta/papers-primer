"""Show PPO's clipped surrogate for one positive-advantage action."""

# Reading guide: follow the named helpers in data-flow order, then inspect the
# assertions at the bottom. Change one toy input at a time and rerun the file.
from __future__ import annotations
def clip(x:float,lo:float,hi:float)->float:return max(lo,min(hi,x))
def main()->None:
 ratio,advantage,epsilon=1.35,2.0,.2
 unclipped=ratio*advantage; clipped=clip(ratio,1-epsilon,1+epsilon)*advantage
 objective=min(unclipped,clipped)
 print(f'unclipped={unclipped:.2f}, clipped={clipped:.2f}, surrogate={objective:.2f}')
 assert objective==clipped and objective<unclipped
 print('ok: clipping removes incentive for an overly large policy increase')
if __name__=='__main__':main()
