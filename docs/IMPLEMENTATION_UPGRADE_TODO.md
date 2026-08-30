# Paper-Faithful Implementation Upgrade Checklist

**Standard:** a CPU-runnable compact implementation of the paper's real core
algorithm, with a purpose-focused module docstring, function-level
explanations, comments for non-obvious steps, and assertions for a meaningful
paper invariant. These are not intended to reproduce original training scale.

Completed: 08 RoPE, 09 DPO.

## Transformer and language models

- [ ] 01 Attention Is All You Need — encoder/decoder Transformer block and masks
- [x] 02 BERT — bidirectional encoder with MLM batching and loss
- [x] 03 GPT-3 — causal decoder, positional embeddings, in-context prompt
- [x] 04 LoRA — frozen linear layer plus trainable low-rank adapters
- [x] 05 InstructGPT — reward model, PPO-style policy update, KL control
- [ ] 06 Chinchilla — fitted scaling-law compute allocation sweep
- [x] 07 FlashAttention — tiled forward and causal/masked case
- [x] 10 Switch Transformer — top-1 MoE dispatch, capacity, balancing loss
- [x] 11 SentencePiece — unigram model scoring and Viterbi decode
- [x] 12 T5 — paper-faithful span corruption and sentinel-target construction
- [x] 13 RAG — retriever scoring, top-k documents, marginal generation probability
- [x] 14 Chain-of-Thought — trace parsing and self-consistency aggregation
- [x] 15 PagedAttention — logical/physical KV blocks, sharing, copy-on-write
- [x] 16 word2vec — skip-gram minibatches and negative-sampling objective
- [x] 17 Adam — multi-step moments, bias correction, parameter update
- [ ] 30 Scaling Laws — regression fit and held-out compute extrapolation

## Vision and generative models

- [ ] 18 ResNet — residual stage, downsample shortcut, short classifier step
- [ ] 19 GAN — generator/discriminator alternating optimization
- [ ] 20 CLIP — dual encoders, normalized similarities, symmetric contrastive loss
- [ ] 21 VAE — encoder, reparameterization, decoder, ELBO training step
- [x] 22 Batch Normalization — train/eval running statistics and affine transform
- [ ] 23 U-Net — contracting path, skip joins, expanding path, segmentation loss
- [ ] 25 Vision Transformer — patch embed, class token, Transformer encoder
- [ ] 27 SimCLR — augmentations, encoder/projection head, NT-Xent loss
- [ ] 29 DDPM — forward diffusion, noise predictor, reverse sampling step

## Reinforcement learning and graphs

- [ ] 24 PPO — rollout advantages, clipped surrogate, value and entropy losses
- [ ] 26 DQN — replay buffer, target network, Bellman update
- [ ] 28 Graph Attention Networks — multi-head attention, adjacency masking, layer

## Completion checks

- [ ] Run every script under `implementations/*/code/`
- [ ] Compile all Python files and run `git diff --check`
- [ ] Update the implementation index/readme if an interface changes
- [ ] Commit and push the completed upgrade set
