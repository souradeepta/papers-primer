# papers-primer

Explainers for foundational AI/ML papers, written for both a CS student
and a software engineer with industry experience. See `SPEC.md` for the
per-paper requirements and `templates/PAPER_TEMPLATE.md` to add a new one.

| # | Paper | arXiv | Status |
|---|-------|-------|--------|
| 01 | [Attention Is All You Need](papers/01-attention-is-all-you-need/README.md) | [1706.03762](https://arxiv.org/abs/1706.03762) | done |
| 02 | [BERT](papers/02-bert/README.md) | [1810.04805](https://arxiv.org/abs/1810.04805) | done |
| 03 | [GPT-3: Language Models are Few-Shot Learners](papers/03-gpt3-few-shot-learners/README.md) | [2005.14165](https://arxiv.org/abs/2005.14165) | done |
| 04 | [LoRA: Low-Rank Adaptation](papers/04-lora/README.md) | [2106.09685](https://arxiv.org/abs/2106.09685) | done |
| 05 | [InstructGPT (RLHF)](papers/05-instructgpt-rlhf/README.md) | [2203.02155](https://arxiv.org/abs/2203.02155) | done |
| 06 | [Chinchilla: Training Compute-Optimal Large Language Models](papers/06-chinchilla/README.md) | [2203.15556](https://arxiv.org/abs/2203.15556) | done |
| 07 | [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](papers/07-flashattention/README.md) | [2205.14135](https://arxiv.org/abs/2205.14135) | done |
| 08 | [RoFormer: Enhanced Transformer with Rotary Position Embedding](papers/08-roformer-rope/README.md) | [2104.09864](https://arxiv.org/abs/2104.09864) | done |
| 09 | [Direct Preference Optimization](papers/09-dpo/README.md) | [2305.18290](https://arxiv.org/abs/2305.18290) | done |
| 10 | [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](papers/10-switch-transformer/README.md) | [2101.03961](https://arxiv.org/abs/2101.03961) | done |
| 11 | [SentencePiece](papers/11-sentencepiece/README.md) | [1808.06226](https://arxiv.org/abs/1808.06226) | done |
| 12 | [T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](papers/12-t5/README.md) | [1910.10683](https://arxiv.org/abs/1910.10683) | done |
| 13 | [Retrieval-Augmented Generation](papers/13-rag/README.md) | [2005.11401](https://arxiv.org/abs/2005.11401) | done |
| 14 | [Chain-of-Thought Prompting](papers/14-chain-of-thought/README.md) | [2201.11903](https://arxiv.org/abs/2201.11903) | done |
| 15 | [PagedAttention: Efficient Memory Management for Large Language Model Serving](papers/15-pagedattention-vllm/README.md) | [2309.06180](https://arxiv.org/abs/2309.06180) | done |

## Adding a new paper
1. Copy `templates/PAPER_TEMPLATE.md` to `papers/NN-slug/README.md`.
2. Fill in every section per `SPEC.md`.
3. Add `code/*.py` (runnable) and `assets/*.gif` (self-generated).
4. Run `pytest scripts/validate_paper.py` until green.
5. Add a row to the table above.
