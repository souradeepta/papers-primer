"""PagedAttention-style KV-cache block allocation and copy-on-write sharing.

vLLM maps each sequence's logical token blocks to non-contiguous physical
KV-cache blocks, analogous to virtual memory. Shared prompts initially share
physical blocks; appending to a shared final block requires copy-on-write.
This CPU model implements those allocation and lifetime rules, not attention
kernel math.
"""

from __future__ import annotations


class KVBlockManager:
    """Manage physical KV blocks and per-sequence logical block tables."""

    def __init__(self, block_size: int = 4) -> None:
        self.block_size, self._next_id = block_size, 0
        self.references: dict[int, int] = {}

    def allocate(self) -> int:
        """Allocate a physical block with one owning sequence reference."""
        block = self._next_id
        self._next_id += 1
        self.references[block] = 1
        return block

    def fork_prefix(self, table: list[int]) -> list[int]:
        """Create a new logical table sharing every immutable prefix block."""
        for block in table:
            self.references[block] += 1
        return list(table)

    def append(self, table: list[int], token_count: int) -> tuple[int, int]:
        """Return physical block/offset for a new token, copying shared tails."""
        logical_block, offset = divmod(token_count, self.block_size)
        if logical_block == len(table):
            table.append(self.allocate())
        elif self.references[table[-1]] > 1:
            # Copy-on-write avoids changing cached keys/values visible to fork.
            old = table[-1]
            self.references[old] -= 1
            table[-1] = self.allocate()
        return table[logical_block], offset

    def release(self, table: list[int]) -> None:
        """Drop a sequence and reclaim a physical block only at refcount zero."""
        for block in table:
            self.references[block] -= 1
            if self.references[block] == 0:
                del self.references[block]


def main() -> None:
    manager = KVBlockManager(block_size=4)
    prompt = [manager.allocate(), manager.allocate()]
    branch = manager.fork_prefix(prompt)
    physical, offset = manager.append(branch, token_count=7)
    print(f"shared prefix refs: {manager.references}; appended location: ({physical}, {offset})")
    assert prompt[1] != branch[1] and offset == 3
    manager.release(prompt)
    assert physical in manager.references
    manager.release(branch)
    assert not manager.references
    print("ok: forked prompts share blocks, writes copy shared tails, and refcounts reclaim memory")


if __name__ == "__main__":
    main()
