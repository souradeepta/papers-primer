"""Normalize two graph-neighbor attention scores and aggregate their features."""

# Reading guide: follow the named helpers in data-flow order, then inspect the
# assertions at the bottom. Change one toy input at a time and rerun the file.
from __future__ import annotations
import math
def main()->None:
 scores=[2.0,0.0]; exps=[math.exp(x) for x in scores]; weights=[x/sum(exps) for x in exps]
 features=[1.0,5.0]; aggregate=sum(w*x for w,x in zip(weights,features))
 print(f'weights={weights}; aggregate={aggregate:.3f}')
 assert abs(sum(weights)-1)<1e-12 and weights[0]>weights[1] and 1<aggregate<5
 print('ok: attention weights form a neighborhood distribution')
if __name__=='__main__':main()

