"""Compute one replayed Q-learning target without a neural-network dependency."""

# Reading guide: follow the named helpers in data-flow order, then inspect the
# assertions at the bottom. Change one toy input at a time and rerun the file.
from __future__ import annotations
def main()->None:
 reward,gamma,next_q,terminal=1.0,.99,[2.0,3.0],False
 target=reward if terminal else reward+gamma*max(next_q)
 print(f'TD target={target:.2f}')
 assert abs(target-3.97)<1e-9 and reward==1.0
 print('ok: nonterminal target bootstraps from the best next-action value')
if __name__=='__main__':main()
