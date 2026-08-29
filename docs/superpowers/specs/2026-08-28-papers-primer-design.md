# papers-primer — Design Spec

Date: 2026-08-28
Status: approved (chat), pending user review of this file

## Goal

A growing collection of markdown explainers for foundational AI/ML papers
(transformers, LLMs), each written to serve two audiences at once — a CS
student building first-principles understanding, and a software engineer
with industry experience who wants the practical/production angle — with
diagrams and self-generated GIFs for anything better shown than described.

First batch: 5 papers.
1. `01-attention-is-all-you-need` — Vaswani et al. 2017
2. `02-bert` — Devlin et al. 2018
3. `03-gpt3-few-shot-learners` — Brown et al. 2020
4. `04-lora` — Hu et al. 2021
5. `05-instructgpt-rlhf` — Ouyang et al. 2022

## Out of scope (this spec)

- No website/hosting — plain markdown, GitHub-renderable.
- No automated paper-ingestion pipeline (fetching/parsing PDFs
  automatically) — adding paper #6+ is a manual, template-guided process.
- No CI. Validation is a local pytest run before each commit.

## Project layout

```
papers-primer/
  README.md                    # index of papers + "how to add a new paper"
  SPEC.md                      # checkable spec, mirrors this design doc's
                                # per-paper requirements section, used by
                                # the validator and by subagent prompts
  templates/PAPER_TEMPLATE.md  # skeleton for a new paper doc
  scripts/
    validate_paper.py          # pytest validator, parametrized over papers/*/
    make_gif_<paper-slug>.py   # one GIF-generation script per paper
  papers/
    01-attention-is-all-you-need/
      README.md                # the actual explainer doc
      assets/                  # *.png, *.gif referenced by README.md
      code/                    # runnable *.py snippet(s) + smoke test
    02-bert/  03-gpt3-few-shot-learners/  04-lora/  05-instructgpt-rlhf/
      (same shape as above)
```

Adding paper #6 later: copy `templates/PAPER_TEMPLATE.md` into a new
`papers/NN-slug/README.md`, follow the per-paper requirements below, run
the validator, add a row to the top-level `README.md` index.

## Per-paper doc requirements (checkable, enforced by `scripts/validate_paper.py`)

Each `papers/NN-slug/README.md` must contain, in this order:

1. **TL;DR** — 3-5 sentences, plain language, no jargon.
2. **Why It Matters** — historical/practical context: what problem existed
   before this paper, what changed after.
3. **Core Intuition** — analogy-driven, no math. Must include ≥1 diagram
   (Mermaid or image).
4. **The Mechanism** — the actual math/architecture, CS-student depth.
   Must include ≥1 Mermaid diagram AND ≥1 GIF (`assets/*.gif`) covering
   the single most motion-worthy dynamic in the paper (e.g. attention
   weights lighting up per token, gradient updates over steps).
5. **Practical Engineering Notes** — SWE depth: production tradeoffs,
   performance/memory implications, where this shows up in real libraries
   (e.g. HF `transformers`, PyTorch source) — named, with pointers.
6. **Runnable Code Example** — a short walkthrough referencing a file in
   `code/`, explaining what it does and what output to expect.
7. **Common Misconceptions & Pitfalls** — ≥2 items.
8. **Interview Q&A** — ≥5 question/answer pairs, the kind an interviewer
   might ask about this paper's ideas.
9. **Further Reading** — ≥3 links (original paper on arXiv, plus related
   work / follow-ups / implementations).

Additional checkable requirements:
- Total prose word count ≥ 2000 (fenced code blocks and Mermaid blocks
  excluded from the count).
- Every image/GIF referenced in the README must exist in `assets/` and be
  non-trivial in size (> 10 KB for GIFs, to catch empty/broken renders).
- `code/` must contain ≥1 `.py` file; running it must exit 0 within 60s
  and must not raise. The script itself should assert on sane output
  shapes/values (e.g. attention output shape matches input shape) —
  the validator checks exit code and traceback-free output, not the
  assertions' content.

## Accuracy requirement

Before writing (or reviewing) a paper's doc, fetch the actual paper
(arXiv abstract page at minimum, ideally the full text) via WebSearch/
WebFetch — do not rely solely on model memory for specific claims (exact
hyperparameters, dataset sizes, ablation results, dates). This applies to
both the Opus-authored paper (01) and every Sonnet-subagent-authored
paper (02-05). Cite the primary source in Further Reading.

## GIF pipeline

One script per paper, `scripts/make_gif_<paper-slug>.py`, using
matplotlib + Pillow writer (no external GPU/model dependency — CPU-only
rendering, since this is illustrative, not data-driven). Each script
renders the one concept flagged in that paper's "The Mechanism" section
and writes to `papers/NN-slug/assets/<name>.gif`. Validator checks the
file exists, is non-trivial size, and is referenced in the README.

## Generation workflow

**Phase 0 — Scaffold** (this session, Opus): create the directory
structure above, `SPEC.md` (derived from this doc's per-paper
requirements section), `templates/PAPER_TEMPLATE.md`, and
`scripts/validate_paper.py`.

**Phase 1 — Baseline paper** (this session, Opus): write paper 01
(Attention Is All You Need) end-to-end — README, code smoke test, GIF
script — as the quality baseline the later subagents are pointed at.
Run the validator on it, iterate until green, commit.

**Phase 2 — Remaining 4 papers** (Sonnet subagents, dispatched only
after explicit user go-ahead per standing rule): one subagent per paper
(02-05), each prompted with the path to `SPEC.md`, `templates/
PAPER_TEMPLATE.md`, and paper 01 as a worked example — not pasted
content, paths only. Each subagent must run the validator against its
own paper before reporting done.

**Phase 3 — Review** (this session, Opus): run the full validator suite
across all 5 papers; separately, read each doc for pedagogical quality
and technical accuracy (not just spec compliance) and fix issues
directly.

**Phase 4 — Index + finish**: write the top-level `README.md` index
(table: title, arXiv link, one-line summary, status), commit per paper
as it lands (checkpoint commits, not one batch), final commit once the
validator is green across all 5.

## Model tiering

Matches this repo's established convention: Opus for planning/scaffolding
and the review gate, Sonnet for the bulk per-paper content generation.
The review pass, not the implementer's model tier, is what's expected to
catch real issues — don't skip it.

## Git / commit conventions

Conventional commits (`feat:`/`docs:`/etc.), no Claude/Anthropic
attribution trailer, checkpoint commits between phases and per paper
rather than one batch at the end. Repo kept private.
