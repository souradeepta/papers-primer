# Runnable Python Implementations

This directory is the top-level home for the collection's 30 small,
standalone Python implementations. Each program isolates one central idea
from its paper rather than attempting to reproduce the full training system.
That makes it practical to read, modify, and run on a CPU.

Every file begins with a module docstring that states its connection to the
paper. Read the named helpers in data-flow order, then the assertions at the
bottom: each assertion is a deliberately small check of the idea being
demonstrated. Inline comments call out non-obvious tensor shapes, numerical
choices, and paper-specific invariants.

## Setup and use

The examples use only the Python standard library unless their source imports
`torch`. For the PyTorch examples, install a CPU build appropriate for your
platform:

```bash
python3 -m pip install torch
```

From the repository root, run a program with its exact path. For example:

```bash
python3 implementations/01-attention-is-all-you-need/code/attention_from_scratch.py
```

Successful examples print a short `ok:` message or a computed toy result and
exit with status zero. Change one toy input at a time, rerun, and use the
assertion failure (if any) to understand which property the paper relies on.
These are teaching implementations, not production models or faithful
training reproductions.

## Index

| # | Paper | Implementation |
|---|---|---|
| 01 | Attention Is All You Need | [attention](01-attention-is-all-you-need/code/attention_from_scratch.py) |
| 02 | BERT | [masked-language modeling](02-bert/code/bert_mlm_from_scratch.py) |
| 03 | GPT-3 | [in-context decoder](03-gpt3-few-shot-learners/code/gpt3_incontext_decoder.py) |
| 04 | LoRA | [low-rank adaptation](04-lora/code/lora_from_scratch.py) |
| 05 | InstructGPT / RLHF | [preference optimization](05-instructgpt-rlhf/code/rlhf_from_scratch.py) |
| 06 | Chinchilla | [compute-optimal scaling](06-chinchilla/code/compute_optimal_scaling.py) |
| 07 | FlashAttention | [streaming attention](07-flashattention/code/flash_attention_demo.py) |
| 08 | RoFormer / RoPE | [rotary positions](08-roformer-rope/code/rope_relative_position.py) |
| 09 | DPO | [preference loss](09-dpo/code/dpo_toy_preference.py) |
| 10 | Switch Transformer | [expert routing](10-switch-transformer/code/switch_routing_demo.py) |
| 11 | SentencePiece | [unigram segmentation](11-sentencepiece/code/unigram_segmentation.py) |
| 12 | T5 | [span corruption](12-t5/code/span_corruption.py) |
| 13 | RAG | [retrieval marginalization](13-rag/code/retrieval_marginalization.py) |
| 14 | Chain-of-Thought | [trace voting](14-chain-of-thought/code/trace_majority_vote.py) |
| 15 | PagedAttention | [block management](15-pagedattention-vllm/code/block_manager.py) |
| 16 | word2vec | [negative sampling](16-word2vec/code/skipgram_negative_sampling.py) |
| 17 | Adam | [optimizer step](17-adam/code/adam_step.py) |
| 18 | ResNet | [residual block](18-resnet/code/residual_block.py) |
| 19 | GAN | [adversarial step](19-gan/code/adversarial_step.py) |
| 20 | CLIP | [contrastive ranking](20-clip/code/contrastive_ranking.py) |
| 21 | VAE | [reparameterization](21-vae/code/reparameterization.py) |
| 22 | Batch Normalization | [batch normalization](22-batch-normalization/code/batch_norm.py) |
| 23 | U-Net | [skip concatenation](23-unet/code/skip_concat.py) |
| 24 | PPO | [clipped objective](24-ppo/code/clipped_objective.py) |
| 25 | Vision Transformer | [patch tokens](25-vision-transformer/code/patch_tokens.py) |
| 26 | DQN | [TD target](26-dqn/code/td_target.py) |
| 27 | SimCLR | [contrastive pair](27-simclr/code/contrastive_pair.py) |
| 28 | Graph Attention Networks | [neighbor attention](28-graph-attention-networks/code/neighbor_attention.py) |
| 29 | DDPM | [noise schedule](29-ddpm/code/noise_schedule.py) |
| 30 | Scaling Laws | [power law](30-scaling-laws/code/power_law.py) |
| 31 | Long Short-Term Memory | [gated LSTM cell](31-long-short-term-memory/code/lstm_cell.py) |
| 32 | Sequence to Sequence Learning | [LSTM encoder-decoder](32-sequence-to-sequence-learning/code/seq2seq_lstm.py) |
| 33 | Bahdanau Attention | [additive attention](33-bahdanau-attention/code/additive_attention.py) |
| 34 | Dropout | [training/inference behavior](34-dropout/code/dropout_training.py) |
| 35 | GloVe | [weighted least squares](35-glove/code/glove_weighted_least_squares.py) |

The matching prose explainer, diagrams, and source links remain under
`papers/NN-slug/`. The two locations intentionally keep reading material
separate from runnable code.
