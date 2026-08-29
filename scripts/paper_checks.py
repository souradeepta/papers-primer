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
