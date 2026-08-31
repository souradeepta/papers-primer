# Cold-Start Guide for Maintainers

Start at the repository root. This is a 35-paper collection; every paper is
accountable to the same `SPEC.md` contract, including the later papers.

## First commands

```bash
git status --short
python3 -m pytest -q
for d in papers/*; do python3 scripts/validate_paper.py "$d" || exit 1; done
```

Read `SPEC.md`, `docs/LEARNINGS.md`, and `templates/PAPER_TEMPLATE.md` before
editing. Use the existing generators in `scripts/` when changing collection-wide
content so all pages receive the same treatment.

Also read [`LICENSE`](../LICENSE) and [`DISCLOSURES.md`](../DISCLOSURES.md)
before publishing. Preserve the MIT terms, source attribution, educational
disclaimer, AI-assistance disclosure, non-affiliation statement, and
third-party dependency notice.

## Content review checklist

- Four learner-first sections are concrete and paper-specific.
- Mechanism contains an equation, Mermaid diagram, GIF, and implementation path.
- Pitfalls has at least four explained misconceptions or failure traps.
- Quick Concept Checks has at least six explanatory answers, not one-liners.
- Interview Q&A has paragraph-length SDE2 answers and explicit follow-ups.
- Published pages contain reader-facing prose only; no authoring instructions.

When a later paper is weaker, fix the content and strengthen the validator so
the same weakness cannot pass again. Keep Markdown semantic and portable:
use fenced code, `inline code`, **bold** labels, and accessible blockquotes;
avoid relying on font colors.

## Final release gate

```bash
python3 -m pytest -q
for d in papers/*; do python3 scripts/validate_paper.py "$d" || exit 1; done
git diff --check
git status --short
```

Confirm the worktree contains only intended files, then commit and push to
`origin/master`. Verify the remote branch includes the commit after pushing.
