"""Pure check functions for papers-primer's paper-doc validator."""
import re
import subprocess
import sys
from pathlib import Path

REQUIRED_SECTIONS = [
    "TL;DR",
    "Fun Map for First Years",
    "Math Playground",
    "Background: What Came Before",
    "Why It Matters",
    "Core Intuition",
    "The Mechanism",
    "Practical Engineering Notes",
    "Runnable Code Example",
    "Common Misconceptions & Pitfalls",
    "Interview Q&A",
    "Further Reading",
]
CS_ANALOGY_MARKER = "💻 **CS analogy:**"

_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+\.gif)\)")
_QA_RE = re.compile(r"\*\*Q:\*\*")
_FOLLOWUP_RE = re.compile(r"\*\*Follow-up:\*\*")
_ANSWER_RE = re.compile(r"\*\*A:\*\*(.*?)(?=\n\n\*\*(?:Q|Follow-up):|\Z)", re.DOTALL)
_LINK_RE = re.compile(r"\[[^\]]+\]\((https?://[^)]+)\)")
_GENERIC_INTERVIEW_PHRASES = (
    "Start by identifying the data structure entering the operation",
    "Assert the property that makes the method meaningful",
    "Reject it when it changes the evaluation contract",
    "Reproduce the smallest production-shaped input and compare intermediate values",
    "Show one minimal failing example, the expected invariant, the observed intermediate divergence",
)
_QUICK_ANSWER_RE = re.compile(r"\*\*A:\*\*\s*(.*?)(?=\n\n\*\*Q:|\Z)", re.DOTALL)


def check_sections(text: str) -> list[str]:
    missing = [s for s in REQUIRED_SECTIONS if _section_body(text, s) is None]
    if CS_ANALOGY_MARKER not in text:
        missing.append("CS analogy")
    if missing:
        return missing

    ordered_markers = [
        "## TL;DR",
        "## Fun Map for First Years",
        CS_ANALOGY_MARKER,
        "## Math Playground",
        "## Background: What Came Before",
        "## Why It Matters",
        "## Core Intuition",
        "## The Mechanism",
        "## Practical Engineering Notes",
        "## Runnable Code Example",
        "## Common Misconceptions & Pitfalls",
        "## Interview Q&A",
        "## Further Reading",
    ]
    positions = [section_position(text, marker) if marker.startswith("## ") else text.index(marker) for marker in ordered_markers]
    if positions != sorted(positions):
        return ["sections are not in the required learner-first order"]
    return []


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
    """Run a paper's canonical top-level implementation, or legacy local code."""
    code_dir = paper_dir / "code"
    canonical_dir = paper_dir.parent.parent / "implementations" / paper_dir.name / "code"
    if not code_dir.exists() and canonical_dir.exists():
        code_dir = canonical_dir
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


_SECTION_RE = re.compile(r"^## (.+)$", re.MULTILINE)


def _section_body(text: str, heading: str) -> str | None:
    """Return the text between `## {heading}` and the next `## ` heading (or EOF)."""
    matches = list(_SECTION_RE.finditer(text))
    for i, m in enumerate(matches):
        actual = re.sub(r"^(?:\d+|[A-Z])[.)]\s+", "", m.group(1).strip())
        if actual == heading:
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            return text[start:end]
    return None


def section_position(text: str, marker: str) -> int:
    """Find a required h2 while allowing a numeric or alphabetic prefix."""
    heading = marker.removeprefix("## ")
    match = re.search(rf"^##\s+(?:(?:\d+|[A-Z])[.)]\s+)?{re.escape(heading)}\s*$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"missing section: {heading}")
    return match.start()


def check_mechanism_mermaid(text: str) -> list[str]:
    body = _section_body(text, "The Mechanism")
    if body is None:
        return ["no '## The Mechanism' section found"]
    if "```mermaid" not in body:
        return ["The Mechanism section has no Mermaid diagram"]
    return []


def check_qa_pairs(text: str) -> int:
    return len(_QA_RE.findall(text))


def check_interview_quality(text: str) -> list[str]:
    """Enforce the paragraph-depth standard in the official Interview Q&A."""
    body = _section_body(text, "Interview Q&A")
    if body is None:
        return ["missing official Interview Q&A section"]
    answers = [" ".join(a.split()) for a in _ANSWER_RE.findall(body)]
    questions = len(re.findall(r"\*\*Q:\*\*", body))
    followups = len(_FOLLOWUP_RE.findall(body))
    errors = []
    if questions < 3:
        errors.append(f"only {questions} scenario questions; need >=3")
    if followups < 3:
        errors.append(f"only {followups} follow-ups; need >=3")
    short = [len(a.split()) for a in answers if len(a.split()) < 40]
    if len(answers) < 6:
        errors.append(f"only {len(answers)} answers; need >=6 including follow-ups")
    if short:
        errors.append(f"{len(short)} answers are shorter than 40 words")
    generic_hits = sum(phrase in body for phrase in _GENERIC_INTERVIEW_PHRASES)
    if generic_hits >= 2:
        errors.append("interview answers contain generic template filler; add paper-specific mechanism, invariant, failure, and test evidence")
    code_terms = re.findall(r"`[^`\n]+`", body)
    if len(code_terms) < 3:
        errors.append("interview section needs at least 3 concrete inline code/equation references")
    return errors


def check_learning_sections(text: str) -> list[str]:
    """Ensure pitfalls and quick checks are explanatory, not placeholders."""
    errors = []
    pitfalls = _section_body(text, "Common Misconceptions & Pitfalls")
    checks = _section_body(text, "Quick Concept Checks")
    if pitfalls is None:
        errors.append("missing Common Misconceptions & Pitfalls section")
    else:
        items = re.findall(r"(?m)^- \*\*.*?\*\*.*?(?=\n- |\Z)", pitfalls, re.DOTALL)
        short_items = [item for item in items if len(item.split()) < 30]
        if len(items) < 4:
            errors.append(f"only {len(items)} misconception/pitfall items; need >=4")
        if short_items:
            errors.append(f"{len(short_items)} misconception/pitfall items are shorter than 30 words")
    if checks is None:
        errors.append("missing Quick Concept Checks section")
    else:
        questions = len(re.findall(r"\*\*Q:\*\*", checks))
        answers = [" ".join(a.split()) for a in _QUICK_ANSWER_RE.findall(checks)]
        if questions < 6:
            errors.append(f"only {questions} quick checks; need >=6")
        short_answers = [a for a in answers if len(a.split()) < 30]
        if short_answers:
            errors.append(f"{len(short_answers)} quick-check answers are shorter than 30 words")
    return errors


def check_runnable_example(text: str) -> list[str]:
    """Require a reproducible, explanatory runnable-example section."""
    body = _section_body(text, "Runnable Code Example")
    if body is None:
        return ["missing Runnable Code Example section"]
    errors = []
    if "```bash" not in body:
        errors.append("runnable example needs an exact fenced bash command")
    if "implementations/" not in body or ".py" not in body:
        errors.append("runnable example must link to a canonical implementation")
    for marker in ("Prerequisites", "Expected behavior", "Production connection"):
        if marker not in body:
            errors.append(f"runnable example missing {marker.lower()} explanation")
    if len(body.split()) < 150:
        errors.append("runnable example is too terse; explain the code, invariant, experiment, and production behavior")
    return errors


def check_further_reading(text: str) -> int:
    return len(_LINK_RE.findall(text))
