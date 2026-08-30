# papers-primer v2 implementation plan

## Scope

Implement the five papers selected in the Batch 2 specification, in this
order: Chinchilla, FlashAttention, RoFormer/RoPE, DPO, and Switch
Transformer. Each paper remains independently useful and must satisfy the
root `SPEC.md` validator before the next one begins.

## Per-paper workflow

1. Fetch and read the primary arXiv source; record exact experimental and
   architectural claims from that source rather than recalling them.
2. Create the paper directory with a 2,000+ word explanation following the
   template exactly. Use an analogy for the intuition, and distinguish the
   paper's result from later practice where relevant.
3. Add a small, CPU-runnable PyTorch program that demonstrates the central
   mechanism and asserts an important invariant.
4. Add a one-off plotting script and generate a GIF from it. The GIF should
   communicate the paper's core trade-off, not serve as decoration.
5. Run the single code example and the full pytest validator. Correct every
   failure before progressing.
6. Do an independent accuracy/spec pass against the primary paper, then
   update the top-level README table and checkpoint the finished paper.

## V2 paper-specific demonstrations

| Paper | Runnable demonstration | GIF |
| --- | --- | --- |
| 06 Chinchilla | Fixed-compute parameter/token allocation | Iso-compute loss valley and its optimum |
| 07 FlashAttention | Tiled online-softmax attention equals reference attention | Tile-by-tile running max/normalizer |
| 08 RoPE | Rotation makes attention scores depend on relative position | Rotating query/key vectors across positions |
| 09 DPO | Preference loss and gradients against a frozen reference | Preferred/dispreferred log-ratio improvement |
| 10 Switch | Top-1 routing, capacity, and auxiliary balancing loss | Token-to-expert routing and load balance |

## Progress

- [x] Scope and paper selection captured in Batch 2 spec.
- [x] 06 — Chinchilla
- [x] 07 — FlashAttention
- [ ] 08 — RoFormer / RoPE
- [ ] 09 — DPO
- [ ] 10 — Switch Transformer

## Global Constraints

These bind every task below and are not restated per-task:

- Root `SPEC.md` is the binding authority: 9 required sections in exact
  order, prose word count >= 2000 (fenced code/Mermaid excluded), a
  ```mermaid fenced block specifically inside "The Mechanism" section
  (not elsewhere), >=1 GIF referenced in "The Mechanism" and present in
  `assets/` at >10 KB, `code/` has >=1 runnable `.py` file that exits 0
  within 60s with no traceback, >=5 Interview Q&A pairs, >=3 Further
  Reading links including the original arXiv abstract page.
- Fetch the actual paper via WebFetch/WebSearch before writing any prose.
  Do not rely on model memory for hyperparameters, dataset sizes, dates,
  or ablation numbers — the arXiv `/abs/<id>` page is the primary source.
- Use `templates/PAPER_TEMPLATE.md` as the section skeleton and
  `papers/06-chinchilla/` as a worked example of the finished shape
  (README.md, code/*.py, assets/*.gif, a `scripts_make_gif.py` one-off
  plotting script kept alongside for reproducibility).
- Validate with `python3 -m pytest scripts/validate_paper.py -k <paper_dir_name>`
  (or the full suite) before reporting DONE — it must pass for every
  paper directory, not just the new one.
- Commit style for this repo: plain Conventional Commits
  (`docs: add <paper> explainer`, `fix: ...`), **no** `Co-Authored-By` or
  `Claude-Session` attribution trailers of any kind. Commit directly to
  `master` — this repo does not use feature branches or worktrees for
  paper additions (see papers 01-06 in `git log`).
- On completion, update the root `README.md` paper table (add the row,
  status `done`) and check the corresponding box in this plan's Progress
  list above.
- The implementer never dispatches subagents (no helpers, no reviewer) —
  review happens after the report, from the controller.

## Task 1: FlashAttention (papers/07-flashattention)

Implement `papers/07-flashattention/` end to end, following the
per-paper workflow above and all Global Constraints.

- Title: "FlashAttention: Fast and Memory-Efficient Exact Attention with
  IO-Awareness"
- Authors: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
- Year: 2022
- arXiv: [2205.14135](https://arxiv.org/abs/2205.14135) — fetch this page
  before writing; do not recall tiling/IO-complexity numbers from memory.
- Directory: `papers/07-flashattention/`
- Positioning note for "Why It Matters": this is the first paper in the
  collection at the hardware/IO level rather than the modeling level —
  exact attention (not an approximation) made fast via GPU
  memory-hierarchy awareness (tiling, online softmax, recomputation in
  the backward pass), not a new architecture or training objective.
  Contrast with sparse/approximate attention variants it displaced as the
  default fast-attention approach.
- Runnable demonstration (`code/`): implement tiled, online-softmax
  attention (block-wise running max and running normalizer, as in the
  paper's Algorithm 1) in plain PyTorch (CPU-only, no custom CUDA
  kernels — the point is the algorithm, not a real IO-aware
  implementation) and assert its output is numerically close
  (`torch.allclose` with a stated tolerance) to standard full
  materialized-attention output on a small random Q/K/V example.
- GIF: visualize the tile-by-tile evolution of the running max and
  running normalizer (or the running output accumulator) as blocks of
  K/V are streamed in — the GIF should communicate *why* streaming
  softmax needs the running-max correction, not just animate for
  decoration.
- Mechanism section must explain: the standard attention memory/IO
  bottleneck (materializing the full N×N score matrix), the online
  softmax trick (rescaling partial sums as new blocks arrive), tiling
  over SRAM-sized blocks, and why recomputation in the backward pass
  trades FLOPs for memory reads instead of storing the N×N matrix.
- Practical Engineering Notes should name real libraries/APIs: e.g.
  `torch.nn.functional.scaled_dot_product_attention` (PyTorch's built-in
  fused/flash path), the `flash-attn` PyPI package, and where FlashAttention
  fits relative to xFormers memory-efficient attention.

Report status (DONE / DONE_WITH_CONCERNS / NEEDS_CONTEXT / BLOCKED),
commits, the validator command and its output summary, and any concerns.

## Task 2: RoFormer / RoPE (papers/08-roformer-rope)

Implement `papers/08-roformer-rope/` end to end, following the
per-paper workflow above and all Global Constraints.

- Title: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- Authors: Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu
- Year: 2021 (revised 2023)
- arXiv: [2104.09864](https://arxiv.org/abs/2104.09864) — fetch this page
  before writing; do not recall the exact rotation-angle formula or
  relative-position derivation from memory.
- Directory: `papers/08-roformer-rope/`
- Positioning note for "Why It Matters": batch 1's Attention Is All You
  Need paper (`papers/01-attention-is-all-you-need/`) used fixed additive
  sinusoidal position embeddings. RoPE instead rotates query/key vectors
  by an angle proportional to absolute position, so the dot product
  between a rotated query and key naturally encodes *relative* position —
  this is what nearly all modern open-weight LLMs (LLaMA, GPT-NeoX,
  Mistral, etc.) actually use. Cross-reference paper 01 explicitly.
- Runnable demonstration (`code/`): implement the RoPE rotation (pairing
  dimensions and applying the rotation-by-position-angle) and assert the
  core relative-position invariant — that `rope(q, pos_i) · rope(k, pos_j)`
  depends only on `pos_i - pos_j`, not on `pos_i` and `pos_j`
  independently (e.g. compute the dot product for several `(pos_i, pos_j)`
  pairs sharing the same offset and assert they match within tolerance,
  and that pairs with a different offset differ).
- GIF: animate query/key vectors (or 2D dimension-pairs of them) rotating
  as position increases, showing visually how the angle between a fixed
  query and a key vector tracks their position difference.
- Mechanism section must explain: the rotation matrix construction
  (pairing consecutive/interleaved dimensions, per-pair frequency
  schedule analogous to the original sinusoidal frequencies), why
  rotation preserves vector norm, and the algebraic identity that makes
  the dot product relative-position-dependent.
- Practical Engineering Notes should name real libraries/usages: e.g.
  HuggingFace `transformers`' rotary embedding implementations (LLaMA,
  GPT-NeoX configs), rotary embedding scaling/interpolation approaches
  used for context-length extension (e.g. position interpolation, NTK-aware
  scaling) as a forward-pointer, without overclaiming RoFormer itself
  introduced those extensions.

Report status (DONE / DONE_WITH_CONCERNS / NEEDS_CONTEXT / BLOCKED),
commits, the validator command and its output summary, and any concerns.

## Task 3: DPO (papers/09-dpo)

Implement `papers/09-dpo/` end to end, following the per-paper workflow
above and all Global Constraints.

- Title: "Direct Preference Optimization: Your Language Model is Secretly
  a Reward Model"
- Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon,
  Christopher D. Manning, Chelsea Finn
- Year: 2023
- arXiv: [2305.18290](https://arxiv.org/abs/2305.18290) — fetch this page
  before writing; do not recall the exact DPO loss formula, the beta
  hyperparameter's role, or benchmark numbers from memory.
- Directory: `papers/09-dpo/`
- Positioning note for "Why It Matters": batch 1's InstructGPT paper
  (`papers/05-instructgpt-rlhf/`) trains a separate reward model then runs
  PPO against it. DPO reframes the same RLHF objective as a single
  closed-form classification loss over preference pairs, using the policy
  itself (and a frozen reference copy) as the implicit reward — no
  separate reward model, no RL loop. Cross-reference paper 05 explicitly
  and be precise about what DPO gives up (no online exploration / KL
  control the way PPO has it) versus what it simplifies away.
- Runnable demonstration (`code/`): implement the DPO loss
  (`-log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))` where
  `log_ratio = log pi_theta(y|x) - log pi_ref(y|x)`) over a small toy
  policy and frozen reference (small random logit tables or a tiny
  linear model are fine — no real LM needed) and demonstrate/assert that
  gradient steps on this loss increase the chosen response's log-prob
  margin over the rejected response's, relative to the frozen reference.
- GIF: show the preferred vs. dispreferred log-ratio (relative to the
  reference model) diverging over training steps as the implicit reward
  margin grows — the trade-off/dynamic the paper's loss is optimizing.
- Mechanism section must explain: the derivation from the RLHF
  KL-constrained reward-maximization objective to the closed-form
  optimal policy, how that substitution turns the Bradley-Terry
  preference model into a loss purely over policy log-probs, and the role
  of beta (implicit KL-penalty strength) and the frozen reference model.
- Practical Engineering Notes should name real usages: e.g. HuggingFace
  `trl`'s `DPOTrainer`, common DPO variants/follow-ups referenced only as
  forward-pointers (e.g. IPO, KTO) without going into their derivations.

Report status (DONE / DONE_WITH_CONCERNS / NEEDS_CONTEXT / BLOCKED),
commits, the validator command and its output summary, and any concerns.

## Task 4: Switch Transformer (papers/10-switch-transformer)

Implement `papers/10-switch-transformer/` end to end, following the
per-paper workflow above and all Global Constraints.

- Title: "Switch Transformers: Scaling to Trillion Parameter Models with
  Simple and Efficient Sparsity"
- Authors: William Fedus, Barret Zoph, Noam Shazeer
- Year: 2021
- arXiv: [2101.03961](https://arxiv.org/abs/2101.03961) — fetch this page
  before writing; do not recall capacity-factor values, auxiliary-loss
  weight, or scaling-experiment numbers from memory.
- Directory: `papers/10-switch-transformer/`
- Positioning note for "Why It Matters": every paper in batch 1 and batch
  2 so far is dense — every token activates every parameter. Switch
  Transformer replaces the FFN in each block with a bank of expert FFNs
  and a top-1 router, so parameter count scales far beyond per-token
  compute. This is the first sparse-model paper in the collection.
- Runnable demonstration (`code/`): implement top-1 token-to-expert
  routing with a fixed per-expert capacity (dropping/overflow tokens past
  capacity, as in the paper) plus the load-balancing auxiliary loss
  (fraction-of-tokens-routed times fraction-of-router-probability, summed
  over experts), on a small toy batch of token embeddings and a small
  number of experts. Assert an invariant such as: the auxiliary loss is
  minimized when routing is uniform across experts, and/or that token
  overflow beyond capacity is correctly tracked and reported.
- GIF: visualize tokens being routed to experts across training
  steps/iterations, and expert load becoming more balanced as the
  auxiliary loss pulls routing away from a collapsed (all-tokens-to-one-expert)
  state.
- Mechanism section must explain: the top-1 (switch) routing decision
  versus top-k routing in earlier MoE work, expert capacity and the
  capacity factor, the load-balancing auxiliary loss and why naive
  top-1 routing collapses without it, and how sparse expert layers keep
  per-token FLOPs roughly constant while total parameters grow.
- Practical Engineering Notes should name real systems: e.g. Mixtral's
  (Mistral AI) sparse MoE layers as a widely-used descendant, GShard/ST-MoE
  follow-ups mentioned only as forward-pointers, and practical concerns
  like expert-parallelism communication cost and load-imbalance-driven
  token dropping in production serving.

Report status (DONE / DONE_WITH_CONCERNS / NEEDS_CONTEXT / BLOCKED),
commits, the validator command and its output summary, and any concerns.
