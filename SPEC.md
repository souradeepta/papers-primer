# papers-primer — Per-Paper Spec

Full design rationale: docs/superpowers/specs/2026-08-28-papers-primer-design.md

## Required sections, in order
1. TL;DR (3-5 sentences, plain language)
2. Why It Matters (context: before/after this paper)
3. Core Intuition (analogy, no math, >=1 diagram)
4. The Mechanism (math/architecture, CS-student depth; >=1 Mermaid diagram AND >=1 GIF)
5. Practical Engineering Notes (SWE depth: prod tradeoffs, perf, named real-library pointers)
6. Runnable Code Example (references a file in code/)
7. Common Misconceptions & Pitfalls (>=2 items)
8. Interview Q&A (>=5 pairs, format: **Q:** ... / **A:** ...)
9. Further Reading (>=3 markdown links, must include the original arXiv paper)

## Checkable numbers
- Prose word count >= 2000 (fenced code/Mermaid blocks excluded)
- The Mechanism section specifically (not just anywhere in the doc) must
  contain a ```mermaid fenced block — a diagram elsewhere (e.g. Core
  Intuition) does not satisfy this
- GIFs referenced must exist in assets/ and be > 10 KB
- code/ has >=1 .py file, exits 0 within 60s, no traceback
- Interview Q&A >= 5 pairs; Further Reading >= 3 links

## Accuracy requirement
Fetch the actual paper (WebFetch/WebSearch) before writing. Do not rely
solely on model memory for specific claims (hyperparameters, dataset
sizes, dates, ablation numbers).

## Directory shape per paper
papers/NN-slug/README.md
papers/NN-slug/assets/*.png *.gif
papers/NN-slug/code/*.py
