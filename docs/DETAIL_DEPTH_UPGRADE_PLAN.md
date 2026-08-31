# Detail-Depth Upgrade Plan

Reference: paper 01, *Attention Is All You Need*. Its strengths are a
step-by-step mechanism, concrete examples, engineering trade-offs, historical
context, paper-specific training details, and diagrams that explain decisions
rather than decorate the page.

## Review result

| Group | Papers | Review finding | Upgrade focus |
|---|---|---|---|
| Reference-quality | 01–05 | Deep mechanism walk-throughs already present | Preserve; only fix factual or clarity defects found during later review |
| Strong but compressed | 06–15 | Good learner sections, but fewer worked mechanics and operational trade-offs | Add paper-specific worked path, failure modes, and concrete implementation decisions |
| Foundation guides | 16–30 | Clear summaries, but mechanisms are usually compressed into a few paragraphs | Add one derivation or worked example, model/data flow, and debugging guidance |
| New sequence batch | 31–35 | Complete but shortest guides in the collection | Add original-paper mechanism details, assumptions, and closer implementation walkthroughs |

## Completion standard

Each upgraded guide must add useful detail in all three places below, without
padding for a numeric target:

1. **Mechanism:** a paper-specific step-by-step data or optimization path,
   including shapes, state, or equations where relevant.
2. **Practice:** concrete training, inference, evaluation, and failure-mode
   guidance tied to that paper's method.
3. **Learning bridge:** a worked intuition or example that links the math and
   diagram to the runnable implementation.

## Execution order

- [x] 06–10: compute-efficient Transformers and alignment
- [x] 11–15: text processing, retrieval, reasoning, and serving
- [x] 16–20: embeddings, optimizers, and vision foundations
- [x] 21–25: generative modeling, normalization, RL, and vision Transformers
- [x] 26–30: RL, contrastive learning, graph attention, diffusion, scaling
- [x] 31–35: recurrent memory, Seq2Seq, additive attention, dropout, GloVe

Every batch receives a prose review, full validator run, implementation run,
and a commit before the next batch begins.
