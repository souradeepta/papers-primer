# papers-primer v6 implementation plan

This batch adds five essential mechanisms not yet represented: value-based deep
control, self-supervised visual representations, graph message passing, modern
diffusion generation, and empirical scaling-law planning. Primary arXiv records
were checked on 2026-08-29.

| # | Paper | arXiv | Gap filled |
| --- | --- | --- | --- |
| 26 | Playing Atari with Deep Reinforcement Learning (DQN) | [1312.5602](https://arxiv.org/abs/1312.5602) | replay-based deep Q-learning from pixels |
| 27 | SimCLR | [2002.05709](https://arxiv.org/abs/2002.05709) | augmentation-driven contrastive visual pretraining |
| 28 | Graph Attention Networks | [1710.10903](https://arxiv.org/abs/1710.10903) | adaptive neighborhood aggregation on graphs |
| 29 | Denoising Diffusion Probabilistic Models | [2006.11239](https://arxiv.org/abs/2006.11239) | iterative denoising generative modeling |
| 30 | Scaling Laws for Neural Language Models | [2001.08361](https://arxiv.org/abs/2001.08361) | compute/data/model-size planning from power laws |

Every explainer will satisfy `SPEC.md`: 1,350+ prose words, ordered sections,
Mermaid mechanism diagram, self-generated GIF over 10 KB, CPU-runnable code,
five Q&A pairs, three further-reading links including primary arXiv, source
checked claims, focused validation, followed by full validation and push.

## Progress

- [x] Selection and plan written.
- [x] 26 — DQN.
- [x] 27 — SimCLR.
- [x] 28 — Graph Attention Networks.
- [x] 29 — DDPM.
- [x] 30 — Scaling Laws.
