# Handoff: continuing papers-primer (Codex)

Date: 2026-08-29. Written for a fresh agent (no prior context on this repo)
picking up where a Claude Code session left off after finishing paper 07.

## What this repo is

`papers-primer` is a collection of long-form explainers for foundational
AI/ML papers, each aimed at both a CS student and an experienced software
engineer. Every paper lives at `papers/NN-slug/` and must satisfy
`SPEC.md` — read that file first, it is the binding spec (exact section
order, word count, diagram/GIF/code/Q&A requirements). `templates/PAPER_TEMPLATE.md`
is the section skeleton to copy.

## Current status

Papers 01-07 are done, committed, and pass validation:

| # | Paper | Status |
|---|-------|--------|
| 01 | Attention Is All You Need | done |
| 02 | BERT | done |
| 03 | GPT-3 | done |
| 04 | LoRA | done |
| 05 | InstructGPT (RLHF) | done |
| 06 | Chinchilla | done |
| 07 | FlashAttention | done |

Verify at any time with:

```
python3 -m pytest scripts/validate_paper.py -v
```

This should show one passing test per paper directory (7 currently).

## What's left

Three papers remain, fully specified in
`docs/superpowers/plans/2026-08-29-papers-primer-v2.md` under **Task 2**,
**Task 3**, and **Task 4** (that file also has a **Global Constraints**
section binding all three — read it, it is not restated here):

- **Task 2 — RoFormer / RoPE** (`papers/08-roformer-rope/`) —
  arXiv:2104.09864. Cross-references paper 01's sinusoidal position
  embeddings.
- **Task 3 — DPO** (`papers/09-dpo/`) — arXiv:2305.18290. Cross-references
  paper 05's RLHF/PPO pipeline.
- **Task 4 — Switch Transformer** (`papers/10-switch-transformer/`) —
  arXiv:2101.03961. First sparse-MoE paper in the collection.

Each task section in the plan gives: exact title/authors/year/arXiv id,
the paper's positioning relative to earlier papers in the collection, the
specific runnable-code demonstration and invariant to assert, the GIF's
intended point, what the Mechanism section must cover, and what real
libraries Practical Engineering Notes should name. Background rationale
for paper selection is in
`docs/superpowers/specs/2026-08-29-papers-primer-batch2-papers.md`.

## Per-paper workflow (apply to Tasks 2, 3, 4 in order)

1. Fetch the primary arXiv source (the `/abs/<id>` page, and the ar5iv
   HTML rendering if the raw PDF fetch degrades to binary) before writing
   any prose. Do not recall hyperparameters, dataset sizes, or ablation
   numbers from memory — this bit repeatedly, cite from the fetch.
2. Write `papers/NN-slug/README.md` following `templates/PAPER_TEMPLATE.md`'s
   9 sections in order. Use `papers/06-chinchilla/README.md` or
   `papers/07-flashattention/README.md` as a worked example of depth and
   tone (both pass validation).
3. Add a small, CPU-runnable PyTorch program in `code/` that demonstrates
   the paper's central mechanism and asserts an invariant (see the task's
   "Runnable demonstration" bullet for what to assert).
4. Add a one-off plotting script (`scripts_make_gif.py`, matching the
   06/07 pattern) that generates a GIF communicating the paper's core
   trade-off — not decoration — into `assets/`. Must be >10KB.
5. Run the code example directly, then the full validator
   (`python3 -m pytest scripts/validate_paper.py -v`). Fix every failure
   before moving on.
6. Do an independent accuracy pass against the primary source (an
   advisor/second-pass review catching mislabeled claims worked well for
   paper 07 — two accuracy bugs were caught this way before commit).
7. Add the paper's row to the root `README.md` table (status `done`) and
   check its box in the Progress list in
   `docs/superpowers/plans/2026-08-29-papers-primer-v2.md`.
8. Commit and push.

## Commit / workflow conventions (binding)

- Plain Conventional Commits (`docs: add <paper> explainer`, `fix: ...`).
  **No** `Co-Authored-By` or similar AI-attribution trailers of any kind —
  this repo's standing convention explicitly forbids them.
- Commit directly to `master`. This repo does not use feature branches or
  worktrees for paper additions (see `git log` — every paper 01-07 landed
  as a direct commit to master).
- Push after each paper, not in one batch at the end.
- Prefer one paper fully done (written, coded, validated, committed,
  pushed) before starting the next, rather than partial work across
  multiple papers in flight at once.

## Open concerns (informational, non-blocking)

- `papers/07-flashattention/README.md`'s Core Intuition section has a
  Mermaid diagram using mechanism-level labels rather than pure-analogy
  language. Flagged as Minor by review, not validator-checked, not
  required to fix — mentioned here only for awareness if you're ever
  touching that file.
- No other known gaps. `python3 -m pytest scripts/validate_paper.py -v`
  is green for all 7 existing papers as of this handoff.

## If you have access to Claude Code's `superpowers` skills

The originating session used the `subagent-driven-development` skill with
a local, git-ignored workspace at `.superpowers/sdd/2026-08-29-papers-primer-v2/`
(ledger, task briefs, review packages) — that workspace is local-only and
not part of this repo's git history, so a fresh clone won't have it. It
is not required to continue this work: the plan file plus `SPEC.md` are
fully self-contained. If you do have that tooling available and want the
same task-brief/review-package workflow, re-run
`scripts/sdd-workspace docs/superpowers/plans/2026-08-29-papers-primer-v2.md`
from that skill to get a fresh local workspace pointed at the same plan.
