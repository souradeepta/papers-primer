# papers-primer v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold the papers-primer repo, build a checkable validator, and produce 5 paper explainer docs (Attention Is All You Need, BERT, GPT-3, LoRA, InstructGPT/RLHF) that each serve both a CS student and an industry SWE.

**Architecture:** A flat `papers/NN-slug/` collection, each with a `README.md` explainer, `assets/` (diagrams + a self-generated GIF), and `code/` (a runnable PyTorch smoke-test snippet). A pytest-based validator (`scripts/validate_paper.py` + `scripts/paper_checks.py`) enforces the checkable requirements from SPEC.md against every paper directory.

**Tech Stack:** Python 3, pytest, PyTorch (CPU), matplotlib + Pillow (GIF rendering), Mermaid (diagrams, GitHub-rendered), plain Markdown.

**Spec:** `docs/superpowers/specs/2026-08-28-papers-primer-design.md`

## Global Constraints

- Word count ≥ 2000 prose words per paper README (fenced code/Mermaid blocks excluded from the count).
- 9 required sections, in order: TL;DR, Why It Matters, Core Intuition, The Mechanism, Practical Engineering Notes, Runnable Code Example, Common Misconceptions & Pitfalls, Interview Q&A, Further Reading.
- ≥1 Mermaid diagram in Core Intuition or The Mechanism; ≥1 GIF in The Mechanism.
- GIF files must exist under `assets/` and be > 10 KB.
- `code/` must contain ≥1 `.py` file that exits 0 within 60s with no traceback.
- Interview Q&A: ≥5 pairs, marked `**Q:**` / `**A:**`. Further Reading: ≥3 markdown links.
- Before writing any paper's content, fetch the actual paper (WebFetch/WebSearch, arXiv at minimum) — do not rely solely on model memory for specific claims.
- No Claude/Anthropic attribution in commits. Conventional commit prefixes (`feat:`/`docs:`/`test:`). Checkpoint commit per task, not one batch.
- **Standing rule:** any subagent dispatch (Task 7-10 below) requires explicit user go-ahead at the time of dispatch, even though this plan is already approved.

---

### Task 1: Repo scaffold — SPEC.md, template, README skeleton

**Files:**
- Create: `SPEC.md`
- Create: `templates/PAPER_TEMPLATE.md`
- Create: `README.md`

**Interfaces:**
- Produces: `SPEC.md` (the source both the validator and later subagent prompts reference by path), `templates/PAPER_TEMPLATE.md` (skeleton later tasks copy).

- [ ] **Step 1: Write `SPEC.md`**

```markdown
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
```

- [ ] **Step 2: Write `templates/PAPER_TEMPLATE.md`**

```markdown
<!-- Copy this file to papers/NN-slug/README.md and fill in every section. -->
<!-- Requirements: SPEC.md (root of this repo). Run scripts/validate_paper.py before committing. -->

# <Paper Title>

## TL;DR
<!-- 3-5 sentences, plain language -->

## Why It Matters
<!-- context: what problem existed before, what changed after -->

## Core Intuition
<!-- analogy-driven, no math. Include at least one diagram (Mermaid or image). -->

## The Mechanism
<!-- math/architecture, CS-student depth. Include >=1 Mermaid diagram AND >=1 GIF: -->
<!-- ![description](assets/name.gif) -->

## Practical Engineering Notes
<!-- SWE depth: production tradeoffs, perf/memory implications, named real-library pointers -->

## Runnable Code Example
<!-- walk through code/name.py: what it does, expected output -->

## Common Misconceptions & Pitfalls
<!-- >=2 items -->

## Interview Q&A
**Q:** <!-- question -->
**A:** <!-- answer -->

<!-- >=5 pairs total -->

## Further Reading
- [Original paper](https://arxiv.org/abs/XXXX.XXXXX)
<!-- >=3 links total -->
```

- [ ] **Step 3: Write `README.md`** (top-level index)

```markdown
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
```

- [ ] **Step 4: Commit**

```bash
git add SPEC.md templates/PAPER_TEMPLATE.md README.md
git commit -m "docs: scaffold papers-primer spec, template, and index"
```

---

### Task 2: TDD the validator's check functions

**Files:**
- Create: `scripts/paper_checks.py`
- Test: `scripts/test_paper_checks.py`

**Interfaces:**
- Produces: `check_sections(text: str) -> list[str]` (returns missing section names, empty if all present), `strip_code_blocks(text: str) -> str`, `count_prose_words(text: str) -> int`, `check_gifs(text: str, paper_dir: Path) -> list[str]` (returns error strings), `check_code_dir(paper_dir: Path) -> list[str]`, `check_qa_pairs(text: str) -> int`, `check_further_reading(text: str) -> int`.
- Consumes: nothing (pure functions + stdlib `re`, `pathlib`, `subprocess`).

- [ ] **Step 1: Write failing tests**

```python
# scripts/test_paper_checks.py
import subprocess
import sys
from pathlib import Path

from paper_checks import (
    check_code_dir,
    check_further_reading,
    check_gifs,
    check_qa_pairs,
    check_sections,
    count_prose_words,
    strip_code_blocks,
)

REQUIRED_SECTIONS = [
    "TL;DR",
    "Why It Matters",
    "Core Intuition",
    "The Mechanism",
    "Practical Engineering Notes",
    "Runnable Code Example",
    "Common Misconceptions & Pitfalls",
    "Interview Q&A",
    "Further Reading",
]


def test_check_sections_all_present():
    text = "\n".join(f"## {s}\ncontent" for s in REQUIRED_SECTIONS)
    assert check_sections(text) == []


def test_check_sections_reports_missing():
    text = "## TL;DR\ncontent"
    missing = check_sections(text)
    assert "Why It Matters" in missing
    assert "TL;DR" not in missing


def test_strip_code_blocks_removes_fences():
    text = "before\n```python\nx = 1\n```\nafter"
    assert strip_code_blocks(text) == "before\n\nafter"


def test_count_prose_words_excludes_code():
    text = "one two three\n```python\nfour five six seven\n```"
    assert count_prose_words(text) == 3


def test_check_gifs_missing_file(tmp_path):
    (tmp_path / "assets").mkdir()
    text = "![attn](assets/attn.gif)"
    errors = check_gifs(text, tmp_path)
    assert any("attn.gif" in e for e in errors)


def test_check_gifs_too_small(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "attn.gif").write_bytes(b"x" * 100)
    text = "![attn](assets/attn.gif)"
    errors = check_gifs(text, tmp_path)
    assert any("too small" in e for e in errors)


def test_check_gifs_passes(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "attn.gif").write_bytes(b"x" * (11 * 1024))
    text = "![attn](assets/attn.gif)"
    assert check_gifs(text, tmp_path) == []


def test_check_code_dir_missing(tmp_path):
    errors = check_code_dir(tmp_path)
    assert any("no .py file" in e for e in errors)


def test_check_code_dir_script_fails(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    (code / "run.py").write_text("raise ValueError('boom')\n")
    errors = check_code_dir(tmp_path)
    assert any("exited" in e or "boom" in e for e in errors)


def test_check_code_dir_script_passes(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    (code / "run.py").write_text("print('ok')\n")
    assert check_code_dir(tmp_path) == []


def test_check_qa_pairs_counts():
    text = "**Q:** a\n**A:** b\n**Q:** c\n**A:** d"
    assert check_qa_pairs(text) == 2


def test_check_further_reading_counts_links():
    text = "- [a](https://x.com)\n- [b](https://y.com)"
    assert check_further_reading(text) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sbisw/github/papers-primer && python -m pytest scripts/test_paper_checks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paper_checks'`

- [ ] **Step 3: Write `scripts/paper_checks.py`**

```python
"""Pure check functions for papers-primer's paper-doc validator."""
import re
import subprocess
import sys
from pathlib import Path

REQUIRED_SECTIONS = [
    "TL;DR",
    "Why It Matters",
    "Core Intuition",
    "The Mechanism",
    "Practical Engineering Notes",
    "Runnable Code Example",
    "Common Misconceptions & Pitfalls",
    "Interview Q&A",
    "Further Reading",
]

_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+\.gif)\)")
_QA_RE = re.compile(r"\*\*Q:\*\*")
_LINK_RE = re.compile(r"\[[^\]]+\]\((https?://[^)]+)\)")


def check_sections(text: str) -> list[str]:
    return [s for s in REQUIRED_SECTIONS if f"## {s}" not in text]


def strip_code_blocks(text: str) -> str:
    return _CODE_FENCE_RE.sub("", text)


def count_prose_words(text: str) -> int:
    return len(strip_code_blocks(text).split())


def check_gifs(text: str, paper_dir: Path) -> list[str]:
    errors = []
    refs = _IMAGE_RE.findall(text)
    if not refs:
        return ["no GIF referenced in README"]
    for ref in refs:
        path = paper_dir / ref
        if not path.exists():
            errors.append(f"referenced GIF not found: {ref}")
            continue
        if path.stat().st_size <= 10 * 1024:
            errors.append(f"referenced GIF too small (<=10KB): {ref}")
    return errors


def check_code_dir(paper_dir: Path) -> list[str]:
    code_dir = paper_dir / "code"
    if not code_dir.exists():
        return ["no code/ directory"]
    py_files = list(code_dir.glob("*.py"))
    if not py_files:
        return ["no .py file in code/"]
    errors = []
    for f in py_files:
        result = subprocess.run(
            [sys.executable, str(f)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            errors.append(f"{f.name} exited {result.returncode}: {result.stderr.strip()[:200]}")
    return errors


def check_qa_pairs(text: str) -> int:
    return len(_QA_RE.findall(text))


def check_further_reading(text: str) -> int:
    return len(_LINK_RE.findall(text))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/sbisw/github/papers-primer && python -m pytest scripts/test_paper_checks.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/paper_checks.py scripts/test_paper_checks.py
git commit -m "test: add TDD'd check functions for the paper validator"
```

---

### Task 3: Integration validator over `papers/*/`

**Files:**
- Create: `scripts/validate_paper.py`

**Interfaces:**
- Consumes: `scripts.paper_checks.{check_sections, count_prose_words, check_gifs, check_code_dir, check_qa_pairs, check_further_reading, REQUIRED_SECTIONS}`
- Produces: a pytest suite runnable as `pytest scripts/validate_paper.py`, parametrized by each `papers/*/README.md` found at collection time.

- [ ] **Step 1: Write `scripts/validate_paper.py`**

```python
"""Integration validator: run all per-paper checks against papers/*/."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from paper_checks import (
    check_code_dir,
    check_further_reading,
    check_gifs,
    check_qa_pairs,
    check_sections,
    count_prose_words,
)

REPO_ROOT = Path(__file__).parent.parent
PAPER_DIRS = sorted((REPO_ROOT / "papers").glob("*/")) if (REPO_ROOT / "papers").exists() else []


@pytest.mark.parametrize("paper_dir", PAPER_DIRS, ids=lambda p: p.name)
def test_paper_is_spec_compliant(paper_dir: Path):
    readme = paper_dir / "README.md"
    assert readme.exists(), f"{paper_dir.name}: missing README.md"
    text = readme.read_text()

    missing = check_sections(text)
    assert not missing, f"{paper_dir.name}: missing sections {missing}"

    words = count_prose_words(text)
    assert words >= 2000, f"{paper_dir.name}: {words} prose words, need >= 2000"

    gif_errors = check_gifs(text, paper_dir)
    assert not gif_errors, f"{paper_dir.name}: {gif_errors}"

    code_errors = check_code_dir(paper_dir)
    assert not code_errors, f"{paper_dir.name}: {code_errors}"

    qa_count = check_qa_pairs(text)
    assert qa_count >= 5, f"{paper_dir.name}: {qa_count} Q&A pairs, need >= 5"

    reading_count = check_further_reading(text)
    assert reading_count >= 3, f"{paper_dir.name}: {reading_count} further-reading links, need >= 3"
```

- [ ] **Step 2: Run it (expect 0 collected, no papers exist yet)**

Run: `cd /home/sbisw/github/papers-primer && python -m pytest scripts/validate_paper.py -v`
Expected: `no tests ran` (PAPER_DIRS is empty — expected until Task 4)

- [ ] **Step 3: Commit**

```bash
git add scripts/validate_paper.py
git commit -m "test: add integration validator over papers/*/"
```

---

### Task 4: Paper 01 — Attention Is All You Need

**Files:**
- Create: `papers/01-attention-is-all-you-need/README.md`
- Create: `papers/01-attention-is-all-you-need/code/attention_from_scratch.py`
- Create: `papers/01-attention-is-all-you-need/scripts_make_gif.py` (generator, run once, not part of validated code/)
- Create: `papers/01-attention-is-all-you-need/assets/attention_weights.gif`

**Interfaces:**
- Consumes: `SPEC.md`, `templates/PAPER_TEMPLATE.md`
- Produces: a paper directory that passes `pytest scripts/validate_paper.py -k attention`

- [ ] **Step 1: Fetch the primary source**

Fetch `https://arxiv.org/abs/1706.03762` (and the full PDF/HTML if accessible) via WebFetch before writing content — ground specific claims (architecture dims, dataset, BLEU numbers) in the actual paper, not memory.

- [ ] **Step 2: Write `code/attention_from_scratch.py`**

```python
"""Minimal scaled dot-product + multi-head attention, runnable smoke test."""
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        out, _ = scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(out)


if __name__ == "__main__":
    x = torch.randn(2, 5, 16)
    mha = MultiHeadAttention(d_model=16, n_heads=4)
    out = mha(x)
    assert out.shape == x.shape, f"expected {x.shape}, got {out.shape}"
    print(f"ok: output shape {tuple(out.shape)} matches input shape")
```

- [ ] **Step 3: Run the smoke test directly**

Run: `cd /home/sbisw/github/papers-primer && python papers/01-attention-is-all-you-need/code/attention_from_scratch.py`
Expected: `ok: output shape (2, 5, 16) matches input shape`, exit 0

- [ ] **Step 4: Write `scripts_make_gif.py`** (one-off generator, lives in the paper dir, not itself validated)

```python
"""Generate assets/attention_weights.gif: a toy attention-weight heatmap
animating as the query position sweeps across a short sequence."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

np.random.seed(0)
seq_len = 6
tokens = ["The", "cat", "sat", "on", "the", "mat"]
logits = np.random.randn(seq_len, seq_len)

fig, ax = plt.subplots(figsize=(4, 4))
out_path = Path(__file__).parent / "assets" / "attention_weights.gif"
out_path.parent.mkdir(exist_ok=True)

writer = PillowWriter(fps=1)
with writer.saving(fig, str(out_path), dpi=100):
    for query_pos in range(seq_len):
        ax.clear()
        row = logits[query_pos]
        weights = np.exp(row) / np.exp(row).sum()
        ax.bar(range(seq_len), weights, color="#4C72B0")
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(tokens, rotation=45)
        ax.set_ylim(0, 1)
        ax.set_title(f'Attention from "{tokens[query_pos]}"')
        writer.grab_frame()
plt.close(fig)
print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")
```

- [ ] **Step 5: Run the GIF generator**

Run: `cd /home/sbisw/github/papers-primer && python papers/01-attention-is-all-you-need/scripts_make_gif.py`
Expected: prints a byte size > 10240 (10 KB)

- [ ] **Step 6: Write `README.md`** per `SPEC.md` / `templates/PAPER_TEMPLATE.md` — all 9 sections, referencing `assets/attention_weights.gif` and `code/attention_from_scratch.py`, grounded in the fetched paper, ≥2000 prose words, ≥5 Interview Q&A pairs, ≥3 Further Reading links (including the arXiv page fetched in Step 1).

- [ ] **Step 7: Validate**

Run: `cd /home/sbisw/github/papers-primer && python -m pytest scripts/validate_paper.py -k attention -v`
Expected: PASS. If it fails, read the assertion message (it names the exact gap) and fix the README/code/assets, then re-run.

- [ ] **Step 8: Commit**

```bash
git add papers/01-attention-is-all-you-need/
git commit -m "docs: add Attention Is All You Need explainer"
```

---

### Task 5: Update index for paper 01

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Edit the status column** for row 01 from `planned` to `done` in the table written in Task 1 Step 3.
- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: mark Attention Is All You Need as done in index"
```

---

### Task 6: Checkpoint — request go-ahead to dispatch papers 02-05

Not a file-producing task. Before Tasks 7-10, stop and ask the user
explicitly for permission to dispatch 4 Sonnet subagents (one per
remaining paper), per the standing rule against spending usage credits
without explicit per-instance permission. Do not proceed to Task 7 until
that permission is given in this session.

---

### Task 7: Paper 02 — BERT (subagent-dispatched)

**Files:**
- Create: `papers/02-bert/README.md`, `papers/02-bert/code/*.py`, `papers/02-bert/scripts_make_gif.py`, `papers/02-bert/assets/*.gif`

**Interfaces:**
- Consumes: `SPEC.md`, `templates/PAPER_TEMPLATE.md`, `papers/01-attention-is-all-you-need/` as a worked example (paths only, not pasted content).
- Produces: a paper directory passing `pytest scripts/validate_paper.py -k bert`.

- [ ] **Step 1: Dispatch** a fresh subagent with: the paper (BERT, arXiv 1810.04805), and pointers to `SPEC.md`, `templates/PAPER_TEMPLATE.md`, and `papers/01-attention-is-all-you-need/` as the worked example. Instruct it to fetch the primary source before writing, write code/GIF/README, and self-run `pytest scripts/validate_paper.py -k bert` until green before reporting done.
- [ ] **Step 2: Review** the returned diff against `SPEC.md` and for technical accuracy.
- [ ] **Step 3: Run validator**

Run: `cd /home/sbisw/github/papers-primer && python -m pytest scripts/validate_paper.py -k bert -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add papers/02-bert/
git commit -m "docs: add BERT explainer"
```

---

### Task 8: Paper 03 — GPT-3 (subagent-dispatched)

Same structure as Task 7, targeting `papers/03-gpt3-few-shot-learners/`, arXiv 2005.14165, validator filter `-k gpt3`.

- [ ] **Step 1: Dispatch**, **Step 2: Review**, **Step 3: Validate** (`pytest scripts/validate_paper.py -k gpt3 -v`), **Step 4: Commit** (`git commit -m "docs: add GPT-3 explainer"`) — following the exact procedure in Task 7.

---

### Task 9: Paper 04 — LoRA (subagent-dispatched)

Same structure as Task 7, targeting `papers/04-lora/`, arXiv 2106.09685, validator filter `-k lora`.

- [ ] **Step 1: Dispatch**, **Step 2: Review**, **Step 3: Validate** (`pytest scripts/validate_paper.py -k lora -v`), **Step 4: Commit** (`git commit -m "docs: add LoRA explainer"`) — following the exact procedure in Task 7.

---

### Task 10: Paper 05 — InstructGPT / RLHF (subagent-dispatched)

Same structure as Task 7, targeting `papers/05-instructgpt-rlhf/`, arXiv 2203.02155, validator filter `-k instructgpt`.

- [ ] **Step 1: Dispatch**, **Step 2: Review**, **Step 3: Validate** (`pytest scripts/validate_paper.py -k instructgpt -v`), **Step 4: Commit** (`git commit -m "docs: add InstructGPT/RLHF explainer"`) — following the exact procedure in Task 7.

---

### Task 11: Full review pass + final validator run

**Files:**
- Modify: any `papers/*/README.md` with issues found

- [ ] **Step 1: Run the full validator suite**

Run: `cd /home/sbisw/github/papers-primer && python -m pytest scripts/validate_paper.py -v`
Expected: all 5 papers PASS

- [ ] **Step 2: Read each of the 5 READMEs** for pedagogical quality and technical accuracy (not just spec compliance) — cross-check specific claims against the fetched primary sources. Fix issues directly.

- [ ] **Step 3: Re-run validator if any file changed**

Run: `cd /home/sbisw/github/papers-primer && python -m pytest scripts/validate_paper.py -v`
Expected: all 5 papers PASS

- [ ] **Step 4: Commit any fixes**

```bash
git add papers/
git commit -m "docs: fix accuracy/quality issues found in review pass"
```

(Skip if Step 2 found nothing to fix.)

---

### Task 12: Final index update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Edit the status column** for rows 02-05 from `planned` to `done`.
- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: mark all 5 papers-primer v1 papers as done"
```
