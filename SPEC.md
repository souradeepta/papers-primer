# papers-primer — Per-Paper Spec

Full design rationale: docs/superpowers/specs/2026-08-28-papers-primer-design.md

## Required sections, in order
1. TL;DR (3-5 sentences, plain language)
2. Fun Map for First Years (a simple flow diagram and accessible beginner/CS-student explanation)
3. CS analogy (a compact, explicit analogy to a familiar computing idea)
4. Math Playground (the paper's single most essential equation, or central mathematical concept when no equation is appropriate; render it in a fenced text block and explain it in two short paragraphs for a reader with high-school mathematics)
5. Background: What Came Before (two short paragraphs covering the previous approach, its limitation, and why this paper was needed)
6. Why It Matters (context: before/after this paper)
7. Core Intuition (analogy, no math, >=1 diagram)
8. The Mechanism (math/architecture, CS-student depth; >=1 Mermaid diagram AND >=1 GIF)
9. Practical Engineering Notes (SWE depth: prod tradeoffs, perf, named real-library pointers)
10. Runnable Code Example (references a documented file in implementations/NN-slug/code/, with prerequisites, an exact fenced command, expected behavior, a paper-specific invariant, a useful experiment, and a production connection)
11. Common Misconceptions & Pitfalls (>=4 paper-specific items, each with an
    explanation and a practical guard or consequence)
12. Interview Q&A (SDE2 depth: >=3 scenario pairs plus follow-ups; every answer
  must be a paragraph of at least 40 prose words and explain implementation,
  trade-offs, or debugging evidence)
13. Further Reading (>=3 markdown links, must include the original arXiv paper)

The first four learner-first sections are short bridges into the deeper
explanation, not filler. In Math Playground, name the equation or concept,
define its important symbols in ordinary language, and explain what changing
the values does before using specialist vocabulary. Use a CS analogy wherever
it makes the mechanism or math more intuitive. Prefer a tiny diagram that
explains a relationship over a decorative image.

Each learner section must earn its space: include a paper-specific concrete
example, a consequence, or a useful comparison. Do not pad sections with
generic study advice that could describe any paper.

## Checkable numbers
- Prose word count >= 1350 (fenced code/Mermaid blocks excluded). This is
  a quality floor, not a reason to reject an otherwise substantive
  explainer over a few formatting-counted words.
- The Mechanism section specifically (not just anywhere in the doc) must
  contain a ```mermaid fenced block — a diagram elsewhere (e.g. Core
  Intuition) does not satisfy this
- GIFs referenced must exist in assets/ and be > 10 KB
- implementations/NN-slug/code/ has >=1 .py file, exits 0 within 60s, no
  traceback, module docstring, and explanatory comments
- Quick Concept Checks may be concise, but the official Interview Q&A must have
  >=3 scenario pairs, >=3 explicit follow-ups, and >=6 paragraph-length answers
  of at least 40 prose words each. Answers should cover mechanism, production
  trade-offs, failure modes, and testing/debugging evidence at SDE2 level.
- Every answer must be paper-specific. The official section must name the
  paper's equation or data path, state a local invariant, identify a concrete
  failure mode, and prescribe a paper-appropriate test or debugging experiment.
  Generic scaffolding such as “check the inputs and compare outputs” does not
  satisfy this requirement, even when it meets the word count.
- Common Misconceptions & Pitfalls must contain >=4 paper-specific items. Each
  item must explain why the claim is wrong or dangerous and give a consequence,
  invariant, or mitigation. Quick Concept Checks must contain >=6 Q&A pairs;
  every answer must be at least 30 prose words and connect the concept to an
  equation, implementation detail, edge case, or operational decision.
- Runnable Code Example must be reproducible from the repository root and must
  include an exact fenced `bash` command, prerequisites, a link to the canonical
  implementation, expected behavior, the asserted invariant, one experiment,
  and a production connection. A command alone is not documentation.
- Use Markdown semantically: `inline code` for symbols and APIs, fenced
  `text`, `python`, or `mermaid` blocks for runnable or visual material,
  **bold** for labels and invariants, and *italics* sparingly for emphasis.
  Do not depend on arbitrary font colors; they are not portable across GitHub,
  terminals, and screen readers. Prefer blockquotes and bold labels for notes.
- Authoring instructions and quality-gate explanations belong in `SPEC.md`,
  templates, or scripts; they must not be copied into published paper pages.

## Review gate and accountability

Run `python3 scripts/validate_paper.py papers/NN-slug` for each page, or loop
over all `papers/*` directories before merging. A review is incomplete if it
checks only section names or word counts: inspect the rendered Markdown and
sample the final papers for repeated interview prose. The validator must fail
on missing diagrams/GIFs, short interview answers, absent follow-ups, and
generic interview-template filler. New papers inherit this same gate; there
is no lower standard for later entries.
- Further Reading >= 3 links

## Accuracy requirement
Fetch the actual paper (WebFetch/WebSearch) before writing. Do not rely
solely on model memory for specific claims (hyperparameters, dataset
sizes, dates, ablation numbers).

## Directory shape per paper
papers/NN-slug/README.md
papers/NN-slug/assets/*.png *.gif
implementations/NN-slug/code/*.py
