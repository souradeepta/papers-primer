"""Compute a training-mode BatchNorm transform for one scalar feature."""
from __future__ import annotations
def main() -> None:
    xs=[1.0,3.0,5.0,7.0]; mean=sum(xs)/len(xs); var=sum((x-mean)**2 for x in xs)/len(xs)
    ys=[(x-mean)/(var+1e-5)**.5 for x in xs]
    print(f'mean={mean:.1f}, variance={var:.1f}, normalized={ys}')
    assert abs(sum(ys)/len(ys)) < 1e-9 and abs(sum(y*y for y in ys)/len(ys)-1) < 1e-5
    print('ok: training batch is centered and unit variance')
if __name__ == '__main__': main()
