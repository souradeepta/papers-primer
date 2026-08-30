"""SentencePiece-style unigram tokenization with Viterbi decoding.

The unigram language-model variant scores a segmentation as the product of
independent piece probabilities. Real SentencePiece learns and prunes a large
vocabulary with EM; this compact version performs the exact inference step
over a learned-looking probability vocabulary, including the visible
whitespace marker that lets it work without language-specific pretokenizing.
"""

from __future__ import annotations

import math


def normalize(text: str) -> str:
    """Use SentencePiece's visible-space convention so decoding is reversible."""
    return "▁" + text.replace(" ", "▁")


def viterbi_segment(text: str, probabilities: dict[str, float]) -> tuple[list[str], float]:
    """Find argmax segmentation with dynamic programming in negative-log space."""
    costs = {piece: -math.log(probability) for piece, probability in probabilities.items()}
    best: list[tuple[float, list[str]]] = [(math.inf, []) for _ in range(len(text) + 1)]
    best[0] = (0.0, [])
    for end in range(1, len(text) + 1):
        for start in range(end):
            piece = text[start:end]
            if piece not in costs or not math.isfinite(best[start][0]):
                continue
            candidate = best[start][0] + costs[piece]
            if candidate < best[end][0]:
                best[end] = (candidate, best[start][1] + [piece])
    if not math.isfinite(best[-1][0]):
        raise ValueError("vocabulary cannot segment this normalized input")
    return best[-1][1], best[-1][0]


def main() -> None:
    # Values behave like a normalized unigram vocabulary for this toy corpus.
    probabilities = {
        "▁": .04, "hello": .08, "world": .07, "▁hello": .32, "▁world": .31,
        "he": .03, "llo": .02, "wor": .025, "ld": .015,
    }
    encoded = normalize("hello world")
    pieces, negative_log_probability = viterbi_segment(encoded, probabilities)
    fallback = ["▁", "hello", "▁", "world"]
    fallback_cost = sum(-math.log(probabilities[piece]) for piece in fallback)
    decoded = "".join(pieces).replace("▁", " ").lstrip()

    print(f"normalized input: {encoded}")
    print(f"best pieces: {pieces}; negative log probability: {negative_log_probability:.3f}")
    assert decoded == "hello world"
    assert "".join(pieces) == encoded
    assert negative_log_probability < fallback_cost
    print("ok: Viterbi finds a reversible, higher-probability subword segmentation")


if __name__ == "__main__":
    main()
