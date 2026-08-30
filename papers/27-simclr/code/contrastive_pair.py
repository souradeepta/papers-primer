"""Demonstrate that a matching SimCLR view has the highest toy similarity."""
from __future__ import annotations
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def main()->None:
 anchor=[1.,0.]; positive=[.9,.1]; negatives=[[0.,1.],[-1.,0.]]
 scores=[dot(anchor,positive)]+[dot(anchor,n) for n in negatives]
 print('positive and negative similarities:',scores)
 assert scores[0]==max(scores)
 print('ok: the positive augmented view ranks above negatives')
if __name__=='__main__':main()
