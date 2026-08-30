"""Toy T5 span corruption: sentinel-delimited targets reconstruct the input."""
from __future__ import annotations

def corrupt(tokens: list[str], spans: list[tuple[int, int]]):
    source, target, cursor = [], [], 0
    for number, (start, end) in enumerate(spans):
        source.extend(tokens[cursor:start]); sentinel=f'<extra_id_{number}>'
        source.append(sentinel); target.extend([sentinel, *tokens[start:end]]); cursor=end
    source.extend(tokens[cursor:]); target.append(f'<extra_id_{len(spans)}>')
    return source, target

def reconstruct(source: list[str], target: list[str]) -> list[str]:
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
    assert [x for x in source if x.startswith('<extra_id_>')] == []  # malformed marker check
    assert source.count('<extra_id_0>') == 1 and source.count('<extra_id_1>') == 1
    print('ok: ordered sentinels and target spans reconstruct the original sequence')
if __name__ == '__main__': main()
