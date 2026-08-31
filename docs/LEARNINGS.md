# Durable Review Learnings

This file records lessons that must survive a fresh agent or cold start.

## Collection-wide quality lessons

- A page can pass word-count and heading checks while still being shallow. Review
  the rendered content for repeated templates, especially in later batches.
- **Common Misconceptions & Pitfalls** needs explanations, consequences, and
  practical guards. Short claims without the “why” are not useful teaching.
- **Quick Concept Checks** are recall prompts, not flash-card one-liners. Each
  answer should connect the concept to the paper’s equation, implementation,
  edge case, or production decision.
- The official SDE2 interview section must remain paper-specific too: mechanism,
  invariant, trade-off, failure mode, and test evidence should all be visible.
- Authoring instructions belong in `SPEC.md`, templates, and scripts. Never
  leak generation guidance into published paper pages.

## Review procedure

1. Read the binding `SPEC.md` and this file.
2. Inspect several early and late papers for depth parity.
3. Run `python3 -m pytest -q`.
4. Run `python3 scripts/validate_paper.py papers/NN-slug` for every paper.
5. Search for repeated scaffolding and inspect the rendered Markdown.
6. Commit and push only after the full collection is green.
