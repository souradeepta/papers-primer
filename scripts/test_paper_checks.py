import subprocess
import sys
from pathlib import Path

from paper_checks import (
    check_code_dir,
    check_further_reading,
    check_gifs,
    check_interview_quality,
    check_mechanism_mermaid,
    check_qa_pairs,
    check_sections,
    count_prose_words,
    strip_code_blocks,
)

REQUIRED_SECTIONS = [
    "TL;DR",
    "Fun Map for First Years 🧭",
    "Math Playground 🧮",
    "Background: What Came Before 🕰️",
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
    text = "\n".join(
        f"## {section}\ncontent"
        + ("\n💻 **CS analogy:** content" if section == "Fun Map for First Years 🧭" else "")
        for section in REQUIRED_SECTIONS
    )
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


def test_check_interview_quality_rejects_generic_template_filler():
    answer = "This is a deliberately long answer with enough prose to make the paragraph look complete while still failing the substantive review requirement. " * 5
    text = """## Interview Q&A
**Q:** one
**A:** Start by identifying the data structure entering the operation. {answer}

**Follow-up:** two
**A:** Assert the property that makes the method meaningful. {answer}

**Q:** three
**A:** {answer}

**Follow-up:** four
**A:** {answer}

**Q:** five
**A:** {answer}

**Follow-up:** six
**A:** {answer}
""".format(answer=answer)
    errors = check_interview_quality(text)
    assert any("generic template filler" in error for error in errors)


def test_check_further_reading_counts_links():
    text = "- [a](https://x.com)\n- [b](https://y.com)"
    assert check_further_reading(text) == 2


def test_check_mechanism_mermaid_present():
    text = "## The Mechanism\n```mermaid\nflowchart TB\nA-->B\n```\n## Practical Engineering Notes\ncontent"
    assert check_mechanism_mermaid(text) == []


def test_check_mechanism_mermaid_missing():
    text = "## The Mechanism\nno diagram here\n## Practical Engineering Notes\ncontent"
    errors = check_mechanism_mermaid(text)
    assert any("Mermaid" in e for e in errors)


def test_check_mechanism_mermaid_diagram_elsewhere_does_not_count():
    text = (
        "## Core Intuition\n```mermaid\nflowchart TB\nA-->B\n```\n"
        "## The Mechanism\nno diagram here\n## Practical Engineering Notes\ncontent"
    )
    errors = check_mechanism_mermaid(text)
    assert any("Mermaid" in e for e in errors)


def test_check_mechanism_mermaid_missing_section_reports_no_sections_crash():
    # If "## The Mechanism" is absent entirely, report it rather than raising.
    text = "## TL;DR\ncontent"
    errors = check_mechanism_mermaid(text)
    assert any("The Mechanism" in e for e in errors)
