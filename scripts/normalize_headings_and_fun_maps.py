"""Normalize release headings and add a beginner-friendly map walkthrough."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from generate_interview_drilldowns import DATA  # noqa: E402

HEADINGS = [
    ("TL;DR", "1. TL;DR"),
    ("Fun Map for First Years 🧭", "2. Fun Map for First Years"),
    ("Math Playground 🧮", "3. Math Playground"),
    ("Background: What Came Before 🕰️", "4. Background: What Came Before"),
    ("Why It Matters", "5. Why It Matters"),
    ("Core Intuition", "6. Core Intuition"),
    ("The Mechanism", "7. The Mechanism"),
    ("Practical Engineering Notes", "8. Practical Engineering Notes"),
    ("Runnable Code Example", "9. Runnable Code Example"),
    ("Common Misconceptions & Pitfalls", "10. Common Misconceptions & Pitfalls"),
    ("Quick Concept Checks", "11. Quick Concept Checks"),
    ("Interview Q&A", "12. Interview Q&A"),
    ("Further Reading", "13. Further Reading"),
]


def add_map_extension(slug: str, body: str) -> str:
    mechanism, equation, invariant, _, _ = DATA[slug]
    if "### Beginner walkthrough" in body:
        return body
    return body.rstrip() + f'''\n\n### Beginner walkthrough

Read the arrows as a sequence of responsibilities. First identify what enters
the system, then ask what the paper changes, what information is preserved or
discarded, and what leaves the operation. For **{mechanism}**, the key question
is not “does the model sound clever?” but “which intermediate value carries the
new information, and what would go wrong if it were missing?”

### CS student checkpoint

The map corresponds to a small program: input data enters a function, the
paper-specific state or transformation runs, and an assertion checks **{invariant}**.
The equation `{equation}` is the compact specification for that function. Trace
one concrete item through each arrow before thinking about larger batches,
parallel hardware, or production optimizations.\n'''


def main() -> None:
    for path in sorted((ROOT / "papers").glob("*/README.md")):
        text = path.read_text()
        slug = path.parent.name
        for old, new in HEADINGS:
            text = re.sub(rf"^## {re.escape(old)}\s*$", f"## {new}", text, flags=re.MULTILINE)
        start = text.index("## 2. Fun Map for First Years")
        end = text.index("\n## 3. Math Playground", start)
        body = text[start:end]
        text = text[:start] + add_map_extension(slug, body) + text[end:]
        path.write_text(text)
        print(slug)


if __name__ == "__main__":
    main()
