"""Standardize the reader-facing runnable-example section for every paper."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from generate_interview_drilldowns import DATA  # noqa: E402


def code_path(slug: str) -> str:
    files = sorted((ROOT / "implementations" / slug / "code").glob("*.py"))
    if len(files) != 1:
        raise SystemExit(f"expected exactly one canonical code file for {slug}")
    return f"implementations/{slug}/code/{files[0].name}"


def section(slug: str, mechanism: str, equation: str, invariant: str, failure: str, test: str) -> str:
    path = code_path(slug)
    special = "" if slug != "34-dropout" else " The classifier performs a real parameter update before the mode checks, so this is more than a static API demonstration."
    return f'''## Runnable Code Example

### Run from the repository root

Prerequisites: Python 3 and the dependencies imported by [`{path}`]({path}).
The example is intentionally small enough to run on CPU; it is a teaching
implementation, not a production training or serving benchmark.

```bash
python3 {path}
```

### What the example demonstrates

Read the module docstring first, then follow the functions implementing
**{mechanism}**. The program turns `{equation}` into executable operations,
prints a compact result, and checks that **{invariant}**. The assertion matters:
it tests the semantic contract near the mechanism instead of treating a
plausible final number as proof that the implementation is correct.{special}

### Expected behavior and useful experiments

The command should finish without a traceback and print a successful summary
or assertion message. You should observe the paper-specific behavior, not a
particular random numeric value. Change one input at a time: inspect the
intermediate tensor or state, rerun with a boundary case, and then compare the
result with the expected invariant. A useful first experiment is to **{test}**.

### Production connection

The toy program does not model every distributed or large-scale concern. In a
real service, version the preprocessing and configuration, record the relevant
intermediate statistic, and measure peak memory, throughput, p95/p99 latency,
and task quality. The first production guard should target **{failure}**;
preserve a transparent reference path or a canary comparison before replacing
it with a fused, distributed, or highly optimized implementation.

'''


def main() -> None:
    for slug, values in DATA.items():
        path = ROOT / "papers" / slug / "README.md"
        text = path.read_text()
        start = text.index("## Runnable Code Example")
        end_match = re.search(r"\n## Common Misconceptions & Pitfalls\n", text[start:])
        if not end_match:
            raise SystemExit(f"missing pitfalls boundary in {slug}")
        end = start + end_match.start() + 1
        path.write_text(text[:start] + section(slug, *values) + text[end:])
        print(slug)


if __name__ == "__main__":
    main()
