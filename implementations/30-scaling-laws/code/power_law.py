"""Compute a toy power-law loss change when compute doubles."""

# Reading guide: follow the named helpers in data-flow order, then inspect the
# assertions at the bottom. Change one toy input at a time and rerun the file.
from __future__ import annotations
def main()->None:
 baseline,compute,exponent=2.0,1.0,-.1
 doubled=baseline*(2*compute/compute)**exponent
 print(f'loss proxy: {baseline:.3f} -> {doubled:.3f}')
 assert doubled<baseline and doubled>0
 print('ok: a negative power exponent gives diminishing improvement with compute')
if __name__=='__main__':main()

