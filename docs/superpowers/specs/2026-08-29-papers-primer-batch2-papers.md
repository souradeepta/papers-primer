# papers-primer — Batch 2 Papers Spec

Date: 2026-08-29
Status: planning only — no implementation started this session (usage
limits). Picks up where docs/superpowers/specs/2026-08-28-papers-primer-design.md
(batch 1) left off.

## Papers (06-10)

All 5 arXiv IDs below verified via WebFetch against arxiv.org/abs/<id> on
2026-08-29 (title + authors confirmed). No corrections needed.

### 06 — Chinchilla
- Title: "Training Compute-Optimal Large Language Models"
- Authors: Hoffmann, Borgeaud, Mensch, et al. (22 authors, DeepMind)
- Year: 2022
- arXiv: [2203.15556](https://arxiv.org/abs/2203.15556)
- Directory: `papers/06-chinchilla/`
- New territory: scaling laws — how to trade off model size vs. training
  tokens under a fixed compute budget. Batch 1 covered architectures and
  fine-tuning; this is the first paper on *how much to train*, directly
  motivating why GPT-3-era models were undertrained.

### 07 — FlashAttention
- Title: "FlashAttention: Fast and Memory-Efficient Exact Attention with
  IO-Awareness"
- Authors: Dao, Fu, Ermon, Rudra, Ré
- Year: 2022
- arXiv: [2205.14135](https://arxiv.org/abs/2205.14135)
- Directory: `papers/07-flashattention/`
- New territory: systems/kernel-level efficiency — exact attention made
  fast via GPU memory-hierarchy awareness (tiling, recomputation), not a
  new architecture or training objective. First paper in this collection
  at the hardware/IO level rather than the modeling level.

### 08 — RoFormer / RoPE
- Title: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- Authors: Su, Lu, Pan, Murtadha, Wen, Liu
- Year: 2021
- arXiv: [2104.09864](https://arxiv.org/abs/2104.09864)
- Directory: `papers/08-roformer-rope/`
- New territory: positional encoding. Batch 1's Attention paper used fixed
  sinusoidal positions; RoPE's rotary relative-position scheme is what
  nearly all modern LLMs (LLaMA, GPT-NeoX, Mistral, etc.) actually use —
  a load-bearing architectural detail not covered yet.

### 09 — DPO
- Title: "Direct Preference Optimization: Your Language Model is Secretly
  a Reward Model"
- Authors: Rafailov, Sharma, Mitchell, Ermon, Manning, Finn
- Year: 2023
- arXiv: [2305.18290](https://arxiv.org/abs/2305.18290)
- Directory: `papers/09-dpo/`
- New territory: alignment *alternative* — reframes RLHF's reward-model +
  PPO pipeline (batch 1's InstructGPT) as a single closed-form
  classification loss, no separate reward model or RL loop. Direct
  contrast/successor to paper 05.

### 10 — Switch Transformer
- Title: "Switch Transformers: Scaling to Trillion Parameter Models with
  Simple and Efficient Sparsity"
- Authors: Fedus, Zoph, Shazeer
- Year: 2021
- arXiv: [2101.03961](https://arxiv.org/abs/2101.03961)
- Directory: `papers/10-switch-transformer/`
- New territory: sparse scaling — mixture-of-experts routing to scale
  parameter count without scaling per-token compute. Everything in batch 1
  and batch 2 so far is dense; this is the first sparse-model paper.

## Per-paper requirements

Identical to batch 1 — no new conventions. Every paper in this batch
follows `SPEC.md` exactly: all 9 required sections in order, the Mermaid
diagram requirement specifically inside "The Mechanism" (not elsewhere),
≥1 GIF per paper, ≥2000 prose words, runnable code in `code/`, ≥5 Interview
Q&A pairs, ≥3 Further Reading links including the primary arXiv source.
See `SPEC.md` for the exact checkable numbers — not restated here.

## Execution (for the session that implements this)

Same workflow as batch 1: fetch the primary source (WebFetch/WebSearch)
before writing each paper — do not rely on model memory for hyperparameters,
dataset sizes, or ablation numbers. Dispatch via
`superpowers:subagent-driven-development`, one Sonnet implementer per
paper plus a Sonnet reviewer doing combined spec-compliance + technical-
accuracy review per paper (matches batch 1's review-gate finding: the
review pass caught real issues, not the implementer's model tier).
Controller (main session) commits and pushes checkpoint-style, per paper,
not in one batch. Dispatch only after explicit user go-ahead per the
standing subagent-usage rule.

No implementation plan file exists yet — write one via
`superpowers:writing-plans` at the start of the session that picks this
up.

## Status

All 5 papers (06-10): **not started**.
