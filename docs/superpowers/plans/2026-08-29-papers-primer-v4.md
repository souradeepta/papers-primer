# papers-primer v4 implementation plan

## Goal

Add five foundational explainers that fill the collection's largest remaining
cross-cutting gaps: static word representations, optimization, deep visual
backbones, adversarial generation, and language-supervised vision. The papers
were checked against their primary arXiv records on 2026-08-29.

| # | Paper | arXiv | Gap filled |
| --- | --- | --- | --- |
| 16 | Efficient Estimation of Word Representations in Vector Space (word2vec) | [1301.3781](https://arxiv.org/abs/1301.3781) | efficient static distributional embeddings |
| 17 | Adam: A Method for Stochastic Optimization | [1412.6980](https://arxiv.org/abs/1412.6980) | adaptive first-order optimization |
| 18 | Deep Residual Learning for Image Recognition (ResNet) | [1512.03385](https://arxiv.org/abs/1512.03385) | trainable very-deep visual backbones |
| 19 | Generative Adversarial Networks | [1406.2661](https://arxiv.org/abs/1406.2661) | adversarial implicit generation |
| 20 | Learning Transferable Visual Models From Natural Language Supervision (CLIP) | [2103.00020](https://arxiv.org/abs/2103.00020) | contrastive vision-language pretraining |

## Shared implementation standard

Each explainer must meet `SPEC.md`: nine ordered sections, at least 1,350
prose words, a Mermaid diagram in **The Mechanism**, an illustrative generated
GIF over 10 KB, a CPU-only runnable Python demonstration, five Q&A pairs, and
three further-reading links including the primary paper. Re-read the primary
source immediately before prose, avoid unverified exact claims, label all
visuals as illustrative, run the demo plus the focused validator, then update
the root index and this plan. Finish with the full validator and a clean diff.

## Per-paper teaching invariants

1. **word2vec:** implement tiny skip-gram negative-sampling loss and show that
   a positive pair receives a larger update than sampled noise; distinguish the
   original paper's hierarchical-softmax presentation from later popular
   negative sampling.
2. **Adam:** implement scalar/vector Adam updates with bias correction and
   assert the first corrected moments equal the first gradient and its square;
   explain epsilon placement and decoupled weight decay as separate choices.
3. **ResNet:** implement a residual block and assert its identity path passes
   a signal when the residual branch is zero; cover projection shortcuts and
   the difference between training ease and a mathematical guarantee.
4. **GAN:** implement a tiny analytic discriminator/generator game and assert
   the discriminator's real/fake objectives pull in opposite directions;
   explain the non-saturating generator loss as a common later training choice.
5. **CLIP:** implement a small symmetric image-text contrastive loss and assert
   matching pairs rank ahead of mismatched pairs after a hand-built score
   matrix; cover prompt templates and zero-shot classifier construction.

## Progress

- [x] Selection and plan written.
- [x] 16 — word2vec.
- [ ] 17 — Adam.
- [ ] 18 — ResNet.
- [ ] 19 — GAN.
- [ ] 20 — CLIP.
