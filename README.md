# papers-primer

Explainers for foundational AI/ML papers, written for both a CS student
and a software engineer with industry experience. See `SPEC.md` for the
per-paper requirements and `templates/PAPER_TEMPLATE.md` to add a new one.

| # | Paper | arXiv | Status |
|---|-------|-------|--------|
| 01 | Attention Is All You Need | [1706.03762](https://arxiv.org/abs/1706.03762) | planned |
| 02 | BERT | [1810.04805](https://arxiv.org/abs/1810.04805) | planned |
| 03 | GPT-3: Language Models are Few-Shot Learners | [2005.14165](https://arxiv.org/abs/2005.14165) | planned |
| 04 | LoRA: Low-Rank Adaptation | [2106.09685](https://arxiv.org/abs/2106.09685) | planned |
| 05 | InstructGPT (RLHF) | [2203.02155](https://arxiv.org/abs/2203.02155) | planned |

## Adding a new paper
1. Copy `templates/PAPER_TEMPLATE.md` to `papers/NN-slug/README.md`.
2. Fill in every section per `SPEC.md`.
3. Add `code/*.py` (runnable) and `assets/*.gif` (self-generated).
4. Run `pytest scripts/validate_paper.py` until green.
5. Add a row to the table above.
