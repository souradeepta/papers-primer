"""Chain-of-thought prompting with answer extraction and self-consistency.

The original paper shows that few-shot worked examples can induce intermediate
reasoning in sufficiently large language models. Self-consistency, a later
extension, samples multiple reasoning paths and returns their most common
final answer. This script implements the evaluation-side aggregation pipeline;
the traces stand in for model samples so no API or large model is required.
"""

from __future__ import annotations

from collections import Counter
import re


def extract_final_answer(trace: str) -> str:
    """Read a deliberately explicit final-answer field from a generated trace."""
    match = re.search(r"final answer:\s*([^\n.]+)", trace, flags=re.IGNORECASE)
    if not match:
        raise ValueError("trace must end with a 'Final answer:' field")
    return match.group(1).strip()


def self_consistent_answer(traces: list[str]) -> tuple[str, Counter[str]]:
    """Vote over independently sampled chains rather than trusting one path."""
    votes = Counter(extract_final_answer(trace) for trace in traces)
    return votes.most_common(1)[0][0], votes


def main() -> None:
    traces = [
        "3 apples plus 4 apples equals 7. Final answer: 7",
        "Start with 3; add four one-by-one to get 7. Final answer: 7",
        "I accidentally added five. Final answer: 8",
        "The equation is 3 + 4 = 7. Final answer: 7",
        "Counting again gives seven. Final answer: 7",
    ]
    answer, votes = self_consistent_answer(traces)
    print(f"answer votes: {dict(votes)}; selected: {answer}")
    assert answer == "7" and votes["7"] == 4
    print("ok: self-consistency selects the answer supported by independent reasoning paths")


if __name__ == "__main__":
    main()
