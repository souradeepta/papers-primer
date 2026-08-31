# Efficient Memory Management for Large Language Model Serving with PagedAttention

## TL;DR

PagedAttention is a serving-system technique for managing the growing key/value (KV) cache of autoregressive LLM requests. Instead of reserving one contiguous cache region per request, it stores fixed-size KV blocks wherever GPU memory has room and uses a per-request block table to make them logically contiguous. This sharply reduces memory waste from fragmentation and enables safe sharing of prompt-prefix blocks between requests. Kwon et al. build vLLM around this idea and report 2–4× higher throughput at similar latency than the systems compared in their SOSP 2023 evaluation.

## Fun Map for First Years 🧭

PagedAttention stores a model’s growing memory in small blocks, like library books on numbered shelves instead of one giant reserved desk.

`💬 request tokens → 🧱 KV-cache blocks → 🗺️ block table → 🚀 more requests fit`

Long conversations produce a growing cache of past-token information. PagedAttention keeps that cache in small reusable blocks, so one request does not need one giant unbroken memory region.

If two requests begin with the same system prompt, their logical block tables can point to shared physical blocks. They only need separate blocks once the requests diverge.

💻 **CS analogy:** PagedAttention uses virtual-memory-style indirection: a logical token address maps through a table to a physical memory block.

## Math Playground 🧮

The essential equation or rule is:

```text
block = floor(t/B),  offset = t mod B
```

**Essential equation:** block = floor(t/B), offset = t mod B. If each memory block holds B tokens, divide token number t by B: the whole-number part tells you which block to use, and the remainder tells you the slot inside it. For token 23 in blocks of 8, that is block 2, slot 7. This is the same page-number-and-offset arithmetic used by virtual memory.

The whole-number quotient chooses a block and the remainder chooses a slot inside it. For token 23 with blocks of 8, that is block 2, slot 7.

Integer division answers “which page?” and modulo answers “where inside that page?” These operations are constant-time, so indirection adds little computation compared with the memory it saves.

## Background: What Came Before 🕰️

Autoregressive serving stores a growing key–value cache for every request, and conventional contiguous allocation wastes memory when requests have different lengths. That waste restricts batch size and throughput. PagedAttention was needed to manage KV cache memory in fixed blocks, much like virtual memory, so more requests fit safely.

This was needed to serve many variable-length requests without wasting memory on large contiguous allocations.

This let a serving engine handle a changing mix of request lengths with less fragmentation, increasing practical throughput rather than changing model accuracy.

## Why It Matters

FlashAttention makes the attention calculation itself more IO-efficient. Serving has a different bottleneck: after every generated token, the model needs the keys and values for all previous tokens in the request. This KV cache is large, grows token by token, and has an unknown final length when generation begins. A conventional allocator may reserve a maximum-length contiguous region for every request or repeatedly allocate and copy regions as a request grows. Both approaches waste expensive GPU memory and limit how many requests can be batched.

PagedAttention borrows the address-translation idea of operating-system virtual memory. A request sees a logical sequence of KV blocks numbered 0, 1, 2, and so on. The allocator maps each logical block to any available physical GPU block. Attention reads the block table to find the keys and values. The tokens are contiguous *logically* even when physical blocks are scattered. This is an analogy, not CPU paging: vLLM’s blocks remain GPU-resident during normal attention rather than faulting from disk.

The paper’s abstract identifies both fragmentation and redundant duplication as limitations of existing serving systems. Fixed blocks reduce internal waste to at most the partly filled final block; they also make a prompt prefix shareable. Parallel samples or beam branches can reference the same completed prefix blocks and allocate new blocks only when their generated suffixes diverge. Reference counts prevent a shared block from being reclaimed while another request still needs it.

## Core Intuition

Think of a request’s conversation history as a book whose pages need not sit next to one another in a warehouse. The request owns a table saying “logical page 0 is shelf 19, page 1 is shelf 4.” A reader follows the table and experiences one continuous book. A second request with the same opening chapter can point to the same physical pages instead of photocopying them.

The benefit is not that storage becomes infinite. Every generated token still needs KV space. The benefit is that the scheduler can use free blocks from across memory and release them promptly, rather than holding a large reserved but unused tail for each active request. Block size is a trade-off: smaller blocks reduce unused tail space but grow table and management overhead; larger blocks simplify bookkeeping but waste more space in a partially filled final block.

```mermaid
flowchart LR
 R[request logical KV blocks] --> T[block table]
 T --> P[scattered physical GPU blocks]
 P --> A[PagedAttention reads K/V]
 S[shared prompt prefix] --> RC[reference counts]
 RC --> P
```

## The Mechanism

Autoregressive decoding has two phases. During prefill, the server processes the prompt and writes a KV entry for every layer, head, and prompt token. During decode, it produces one token at a time; each step reads the existing KV cache and appends a new entry. The cache therefore grows dynamically and is often the dominant per-request memory cost at long context lengths.

PagedAttention partitions each request’s KV cache into blocks that each hold a fixed number of token positions. A logical token index \(i\) maps to logical block \(\lfloor i/B\rfloor\) and in-block offset \(i\bmod B\), where \(B\) is block capacity. A block table maps that logical block to its physical block ID. The attention kernel gathers keys and values through this mapping while computing attention; it does not require the physical storage to be adjacent.

![A request’s logical KV blocks map to scattered physical blocks, and a shared prefix survives until every reference is released.](assets/paged_kv_blocks.gif)

```mermaid
flowchart TD
 P[prefill writes prompt K/V] --> L[allocate logical block entries]
 L --> F[physical free-block pool]
 F --> BT[request block table]
 D[decode one token] --> BT
 BT --> K[read prior K/V and append new K/V]
 X[branch/shared prefix] --> REF[increment block reference count]
 END[request completion] --> DEC[decrement refs and return zero-ref blocks]
```

Sharing is especially useful for parallel sampling and beam search. A common prompt is encoded once, and branches initially refer to the same physical prefix blocks. If a branch must write into a partially filled shared block, a copy-on-write-style operation is needed before modifying it; otherwise one request could overwrite another’s cache. When a request completes, its references are decremented. Only blocks whose count reaches zero return to the free pool. The toy program asserts this lifetime rule.

PagedAttention is distinct from FlashAttention. FlashAttention changes how attention tiles scores and avoids materializing an \(N\times N\) matrix during an operation. PagedAttention changes where persistent K/V state for many serving requests lives and how it is addressed. A serving stack can use both: a fast attention kernel and a block-managed KV allocator address different layers of the performance problem.

The paper reports near-zero KV-cache waste and flexible sharing as system properties, plus 2–4× throughput gains at the same latency against named systems in its evaluation. Do not treat that range as a universal SLA. Throughput depends on model size, prompt/output lengths, decode algorithm, batch scheduler, GPU, network topology, kernel versions, and workload shape. Longer sequences and complex decoding are specifically noted in the abstract as settings where the reported improvement is more pronounced.

### Mechanism in Code

At implementation level, the mechanism operates on logical token blocks and physical KV pages. A faithful
forward pass should follow this order: append block mappings, read physical pages during decode, share prefixes, and copy on write. Keep the intermediate
representation available while debugging; collapsing everything into one
opaque framework call makes shape and numerical errors much harder to isolate.

The key production failure to guard against is reusing a page after one request frees it while another still references it. Add a tiny
reference test with hand-checkable values, then add a property test that
covers padding, empty/short inputs, boundary probabilities, and the largest
supported shape. Compare intermediate tensors with tolerances appropriate to
the dtype, and log the paper-specific statistic during a canary rollout.


## Practical Engineering Notes

### Worked Math & Dataflow

The compact view below makes the paper's central calculation concrete:

```text
logical → physical blocks
```

In practice, the calculation is a pipeline: A sequence sees contiguous logical token blocks even when its KV cache is stored in non-contiguous physical pages. Reference counting and copy-on-write allow safe prefix sharing. The important engineering
choice is to preserve the paper's intended invariant while making the operation
fit the available memory, batch size, and evaluation protocol.

```mermaid
flowchart LR
    A[paper input] --> B[sequence → block table → KV pages]
    B --> C[paper output]
```

![Animated worked-math walkthrough for PagedAttention](assets/worked_math.gif)


vLLM implements PagedAttention and is the primary real-system pointer. Compare it with Hugging Face Text Generation Inference or other servers at the level of scheduler, batching, cache policy, model support, and operational requirements—not a single headline throughput number. Profile prefill and decode separately: prompt-heavy traffic stresses compute and input bandwidth, while many concurrent decodes stress KV capacity and scheduler behavior.

Monitor free blocks, allocated blocks, partially filled blocks, prefix-cache hits, reference counts, evictions, prefill/decode queue time, and p50/p95/p99 time to first token and inter-token latency. A high cache hit rate can still be bad if permission-aware cache keys are wrong; a full free pool can coexist with poor latency if requests are scheduled inefficiently. Include model revision, tokenizer, adapter identity, and tenant/authorization scope in prefix-cache identity.

Block size is a tuning parameter, not a cosmetic constant. Smaller blocks reduce final-block waste and improve sharing granularity, but add block-table lookup, metadata, and allocation traffic. Larger blocks can improve kernel regularity but reserve more unused token slots at request ends. Test real prompt and completion distributions, including cancellation and streaming disconnects; leaked references after a client disconnect turn a memory-efficiency feature into a slow capacity leak.

Use admission control and quotas. Paged allocation makes memory usage predictable in blocks, so a scheduler can reject or queue work before OOM rather than gambling on maximum output lengths. Release blocks on normal completion, cancellation, error, and worker failure paths. For multi-tenant workloads, never allow a reused physical block to expose stale K/V data; allocator zeroing/isolation behavior and cache-key authorization are security requirements, not optional optimizations.

## Runnable Code Example

### Run it

The implementation is intentionally small and self-checking. From the repository root, use Python 3; the module docstring states the learning goal, comments identify the paper-specific calculation, and assertions verify the toy invariant.

```bash
python3 papers/15-pagedattention-vllm/code/block_manager.py
```

### Read it in order

Start with the module docstring, then follow the named helper calculations and the final assertions. The example is a dependency-light teaching implementation, not a production training system; change one input at a time and rerun it to see which invariant changes.


[`code/block_manager.py`](code/block_manager.py) builds a tiny fixed-size block manager. Two requests share a prefix physical block, map a logical token index to physical block plus offset, then release their blocks in turn. Assertions verify that the prefix remains allocated after the first request finishes and is reclaimed only after the final reference is released.

```bash
python3 papers/15-pagedattention-vllm/code/block_manager.py
```

## Common Misconceptions & Pitfalls

- **“PagedAttention pages KV data to CPU or disk.”** Its central design is GPU-resident fixed blocks with logical-to-physical translation.
- **“It replaces attention kernels such as FlashAttention.”** It manages persistent serving memory; kernels and cache management solve separate problems.
- **“Sharing a prefix is always safe.”** Shared blocks need reference counting, authorization-aware keys, and copy-on-write protection for modifications.
- **“Near-zero waste means no memory limit.”** Active KV entries, weights, scheduler queues, and block metadata still consume finite GPU memory.

## Interview Q&A

**Q:** Why does an LLM server need a KV cache?
**A:** Decode steps reuse prior attention keys and values instead of recomputing the full prefix every token.

**Q:** What does a block table map?
**A:** A request’s logical KV block number to an arbitrary physical GPU block.

**Q:** Why use fixed-size blocks?
**A:** They avoid large contiguous reservations and bound unused tail memory.

**Q:** Why reference count shared prefix blocks?
**A:** A block must not be freed while another beam, sample, or request still reads it.

**Q:** What is the principal operational trade-off?
**A:** Better utilization and sharing versus allocator, scheduler, metadata, and block-size complexity.

## Implementation Walkthrough

PagedAttention stores each request's key-value cache in fixed-size blocks and
maps logical token positions to physical blocks. This avoids reserving one
large contiguous cache for every possible request length and permits sharing
prefix blocks safely. A serving implementation must track block ownership,
reference counts, eviction, and request cancellation; a cache leak or mistaken
shared writable block is both a throughput and correctness failure.

## SDE2 Interview Drill-down

These prompts are designed for a second-level software engineering interview: explain the mechanism, name the operational trade-off, and describe how you would test it.

**Q:** Walk through paged KV-cache management end to end. What does `logical block → physical block` mean in an implementation?
**A:** Start by identifying the data structure entering the operation, the learned or configured values it uses, and the invariant that must hold at the output. In this paper, logical block → physical block is not just notation: it tells you what is compared, normalized, accumulated, or optimized. A strong implementation makes those stages visible in separate functions, keeps tensor shapes and dtypes explicit, and tests a tiny hand-computed example before optimizing. Explain what happens when the inputs are short, padded, empty, or unusually large; those cases often reveal whether the code actually matches the paper.

**Follow-up:** Which invariant would you assert?
**A:** Assert the property that makes the method meaningful: probabilities normalize over valid choices, a residual preserves shape, a target does not bootstrap past termination, or an update leaves frozen state untouched. The assertion should be local and cheap enough to run in tests, not an end-to-end hope such as “accuracy improves.” Also compare the optimized path with a simple reference on random small inputs using an appropriate tolerance. That catches indexing, masking, reduction, and broadcasting errors while the failing example is still understandable.

**Q:** What is the main production trade-off, and how would you capacity-plan it?
**A:** The practical trade-off here is non-contiguous pages reduce fragmentation, while block tables add indirection and bookkeeping. Estimate both arithmetic work and memory movement, then identify whether the service is compute-bound, bandwidth-bound, latency-bound, or limited by coordination. Include batch-size effects, peak activation/state memory, serialization, and cold-start behavior; average throughput can hide a bad tail latency. Choose a baseline configuration, measure it on representative shapes, and document which quality metric is allowed to move. If the system is distributed, include communication and retry behavior rather than treating the model operation as an isolated kernel.

**Follow-up:** What would make you reject an apparently faster optimization?
**A:** Reject it when it changes the evaluation contract, weakens isolation, creates silent quality regressions, or only wins on a synthetic shape. For this paper, watch especially for use-after-free, prefix aliasing, or copy-on-write bugs. A safe rollout uses a reference implementation, shadow traffic or canaries, resource limits, and dashboards for both system and model metrics. Keep the old path available until numerical outputs, error rates, p95/p99 latency, and cost are stable across the important input distributions.

**Q:** How would you debug a model that passes unit tests but fails in production?
**A:** Reproduce the smallest production-shaped input and compare intermediate values against the reference path, not only the final score. Log versioned preprocessing, shapes, masks, random seeds where relevant, and the exact model/configuration identifiers; otherwise a numerical symptom can be caused by data drift or a serving mismatch. Separate failures into data, numerical stability, optimization, and infrastructure categories. For this method, begin with stress concurrent requests with reference-count and isolation assertions, then run a controlled ablation that disables the paper-specific mechanism to determine whether the regression is in the mechanism or its integration.

**Follow-up:** What evidence would you present in the postmortem or interview?
**A:** Show one minimal failing example, the expected invariant, the observed intermediate divergence, and the fix’s regression test. Add a before/after metric table covering quality, memory, throughput, and tail latency, plus the rollout guard that would catch recurrence. This demonstrates engineering judgment: the goal is not merely to identify a clever algorithm, but to make its behavior observable, reproducible, and safe to operate.


## Further Reading

- [Original PagedAttention/vLLM paper](https://arxiv.org/abs/2309.06180)
- [vLLM project](https://github.com/vllm-project/vllm)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [Hugging Face Text Generation Inference](https://github.com/huggingface/text-generation-inference)
