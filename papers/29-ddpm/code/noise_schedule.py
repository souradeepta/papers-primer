"""Apply a one-step diffusion forward noising equation to a scalar sample."""
from __future__ import annotations
import math
def main()->None:
 x0,epsilon,alpha_bar=1.0,-.5,.81
 xt=math.sqrt(alpha_bar)*x0+math.sqrt(1-alpha_bar)*epsilon
 print(f'x_t={xt:.3f}')
 assert abs(xt-(.9-.5*math.sqrt(.19)))<1e-12
 print('ok: forward diffusion mixes signal and Gaussian noise by schedule')
if __name__=='__main__':main()
