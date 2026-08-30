"""Turn a 4x4 toy image into four flattened 2x2 ViT patch tokens."""

# Reading guide: follow the named helpers in data-flow order, then inspect the
# assertions at the bottom. Change one toy input at a time and rerun the file.
from __future__ import annotations
def main()->None:
 image=[[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]; patches=[]
 for row in (0,2):
  for col in (0,2): patches.append([image[row+i][col+j] for i in range(2) for j in range(2)])
 print('patch tokens:',patches)
 assert len(patches)==4 and all(len(p)==4 for p in patches)
 print('ok: non-overlapping patches become a sequence of equal-length tokens')
if __name__=='__main__':main()

