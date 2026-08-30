# papers-primer — Per-Paper Spec

Full design rationale: docs/superpowers/specs/2026-08-28-papers-primer-design.md

## Required sections, in order
1. TL;DR (3-5 sentences, plain language)
2. Fun Map for First Years 🧭 (a simple emoji flow diagram and an accessible explanation)
3. CS analogy (a compact, explicit analogy to a familiar computing idea)
4. Math Playground 🧮 (the paper's key equation or rule, with a plain-language explanation)
5. Background: What Came Before 🕰️ (the previous approach, its limitation, and why this paper was needed)
6. Why It Matters (context: before/after this paper)
7. Core Intuition (analogy, no math, >=1 diagram)
8. The Mechanism (math/architecture, CS-student depth; >=1 Mermaid diagram AND >=1 GIF)
9. Practical Engineering Notes (SWE depth: prod tradeoffs, perf, named real-library pointers)
10. Runnable Code Example (references a file in code/)
11. Common Misconceptions & Pitfalls (>=2 items)
12. Interview Q&A (>=5 pairs, format: **Q:** ... / **A:** ...)
13. Further Reading (>=3 markdown links, must include the original arXiv paper)

The first four learner-first sections are short bridges into the deeper
explanation, not filler. Use a CS analogy wherever it makes the mechanism or
math more intuitive. Prefer a tiny diagram that explains a relationship over a
decorative image.

## Checkable numbers
- Prose word count >= 1350 (fenced code/Mermaid blocks excluded). This is
  a quality floor, not a reason to reject an otherwise substantive
  explainer over a few formatting-counted words.
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
