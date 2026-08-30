"""T5 span-corruption preprocessing with ordered sentinel targets.

T5 turns every NLP task into text-to-text prediction. Its pre-training
objective replaces several contiguous spans with unique sentinel tokens in
the encoder source; the decoder target concatenates those sentinels and their
missing spans. This is the paper's data transformation, kept independent of
a large encoder-decoder model so its invariants are easy to inspect.
"""

# Reading guide: follow the named helpers in data-flow order, then inspect the
# assertions at the bottom. Change one toy input at a time and rerun the file.
from __future__ import annotations

def corrupt(tokens: list[str], spans: list[tuple[int, int]]) -> tuple[list[str], list[str]]:
    """Replace non-overlapping token spans and build the T5 decoder target."""
    source, target, cursor = [], [], 0
    for number, (start, end) in enumerate(spans):
        if not (cursor <= start < end <= len(tokens)):
            raise ValueError("spans must be sorted, non-overlapping, and in range")
        source.extend(tokens[cursor:start]); sentinel=f'<extra_id_{number}>'
        source.append(sentinel); target.extend([sentinel, *tokens[start:end]]); cursor=end
    source.extend(tokens[cursor:]); target.append(f'<extra_id_{len(spans)}>')
    return source, target

def reconstruct(source: list[str], target: list[str]) -> list[str]:
    """Decode sentinel fills into the original sequence; a test-only inverse."""
    fills={}; i=0
    while i < len(target)-1:
        marker=target[i]; i+=1; fill=[]
        while i < len(target) and not target[i].startswith('<extra_id_'):
            fill.append(target[i]); i+=1
        fills[marker]=fill
    return sum((fills.get(token,[token]) for token in source), [])

def main() -> None:
    original='the small cat sat on the warm mat'.split(); spans=[(1,3),(6,7)]
    source,target=corrupt(original,spans); restored=reconstruct(source,target)
    print('source:', ' '.join(source)); print('target:', ' '.join(target))
    assert restored == original
    assert source.count('<extra_id_0>') == 1 and source.count('<extra_id_1>') == 1
    assert target[-1] == '<extra_id_2>' and restored == original
    print('ok: ordered sentinels and target spans reconstruct the original sequence')
if __name__ == '__main__': main()
