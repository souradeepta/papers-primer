"""A tiny Viterbi-style unigram tokenizer showing reversible segmentation."""

# Reading guide: follow the named helpers in data-flow order, then inspect the
# assertions at the bottom. Change one toy input at a time and rerun the file.
from __future__ import annotations
import math

VOCAB = {"▁": 1.7, "hello": 0.2, "world": 0.3, "▁hello": 0.15, "▁world": 0.2,
         "he": 1.2, "llo": 1.3, "wor": 1.4, "ld": 1.2}

def best_segment(text: str, vocab: dict[str, float]) -> tuple[list[str], float]:
    """Return lowest negative-log-score segmentation of a normalized string."""
    dp: list[tuple[float, list[str]]] = [(math.inf, []) for _ in range(len(text)+1)]
    dp[0] = (0.0, [])
    for end in range(1, len(text)+1):
        for start in range(end):
            piece = text[start:end]
            if piece in vocab and dp[start][0] < math.inf:
                score = dp[start][0] + vocab[piece]
                if score < dp[end][0]: dp[end] = (score, dp[start][1] + [piece])
    if dp[-1][0] == math.inf: raise ValueError("no segmentation")
    return dp[-1][1], dp[-1][0]

def main() -> None:
    normalized = "▁hello▁world"  # spaces become the visible whitespace marker
    pieces, score = best_segment(normalized, VOCAB)
    fallback = ["▁", "hello", "▁", "world"]
    fallback_score = sum(VOCAB[p] for p in fallback)
    decoded = "".join(pieces).replace("▁", " ").lstrip()
    print(f"pieces: {pieces}; negative log score: {score:.2f}")
    print(f"decoded: {decoded!r}; fallback score: {fallback_score:.2f}")
    assert "".join(pieces) == normalized and decoded == "hello world"
    assert score <= fallback_score
    print("ok: best-path segmentation is reversible and beats a valid fallback")
if __name__ == "__main__": main()

