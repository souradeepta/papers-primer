"""Integration validator: run all per-paper checks against papers/*/."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from paper_checks import (
    check_code_dir,
    check_further_reading,
    check_gifs,
    check_mechanism_mermaid,
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

    mermaid_errors = check_mechanism_mermaid(text)
    assert not mermaid_errors, f"{paper_dir.name}: {mermaid_errors}"

    code_errors = check_code_dir(paper_dir)
    assert not code_errors, f"{paper_dir.name}: {code_errors}"

    qa_count = check_qa_pairs(text)
    assert qa_count >= 5, f"{paper_dir.name}: {qa_count} Q&A pairs, need >= 5"

    reading_count = check_further_reading(text)
    assert reading_count >= 3, f"{paper_dir.name}: {reading_count} further-reading links, need >= 3"
