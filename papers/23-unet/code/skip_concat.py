"""Illustrate U-Net's same-resolution skip concatenation invariant."""

# Reading guide: follow the named helpers in data-flow order, then inspect the
# assertions at the bottom. Change one toy input at a time and rerun the file.
from __future__ import annotations
def main() -> None:
    encoder=[[1,2],[3,4]]; decoder=[[9,8],[7,6]]
    merged=[a+b for a,b in zip(encoder,decoder)]
    print('merged channels:', merged)
    assert len(merged)==2 and all(len(row)==4 for row in merged)
    print('ok: aligned encoder detail and decoder context concatenate by channel')
if __name__ == '__main__': main()
