"""Tiny symmetric CLIP-style score-matrix ranking check."""

# Reading guide: follow the named helpers in data-flow order, then inspect the
# assertions at the bottom. Change one toy input at a time and rerun the file.
from __future__ import annotations


def main() -> None:
    scores = [[4.0, .5, .1], [.2, 3.5, .4], [.1, .3, 4.2]]
    image_choices = [max(range(3), key=lambda j: row[j]) for row in scores]
    text_choices = [max(range(3), key=lambda i: scores[i][j]) for j in range(3)]
    print('image-to-text:', image_choices, 'text-to-image:', text_choices)
    assert image_choices == [0, 1, 2] and text_choices == [0, 1, 2]
    print('ok: matching pairs rank above mismatches in both directions')


if __name__ == '__main__':
    main()
