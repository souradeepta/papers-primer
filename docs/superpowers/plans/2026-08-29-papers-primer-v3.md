# papers-primer v3 implementation plan

## Goal and selection rationale

Add five independently useful explainers, numbered 11–15, that fill the
largest remaining gaps after papers 01–10: how text reaches a model,
encoder–decoder transfer, external knowledge, inference-time reasoning,
and efficient LLM serving. This is deliberately a broadening batch, not a
second batch of scaling papers. Each selection has an enduring mechanism,
a primary arXiv paper, an executable teaching invariant, and a direct
connection to a system a working ML engineer encounters.

Primary arXiv abstract records (title/authors/year) were checked on
2026-08-29 before this plan was written:

| # | Paper | arXiv | Gap filled |
| --- | --- | --- | --- |
| 11 | SentencePiece | [1808.06226](https://arxiv.org/abs/1808.06226) | raw-text, language-independent subword tokenization |
| 12 | T5 | [1910.10683](https://arxiv.org/abs/1910.10683) | encoder–decoder transfer and text-to-text task interface |
| 13 | RAG | [2005.11401](https://arxiv.org/abs/2005.11401) | retrieval plus parametric generation |
| 14 | Chain-of-Thought Prompting | [2201.11903](https://arxiv.org/abs/2201.11903) | few-shot reasoning traces at inference time |
| 15 | PagedAttention / vLLM | [2309.06180](https://arxiv.org/abs/2309.06180) | KV-cache memory management for serving |

## Global constraints

`SPEC.md` is binding for every task: retain the nine required sections in
order, at least 1,950 prose words, a Mermaid block inside **The
Mechanism**, at least one referenced GIF over 10 KB, a CPU-runnable Python
program in `code/` that exits within 60 seconds, at least five Q&A pairs,
and at least three links in Further Reading including the primary arXiv
abstract page.

For each paper, fetch and read the primary source again immediately before
writing prose. Do not reconstruct exact figures, vocabulary sizes, data
sets, hyperparameters, or ablation claims from this planning document or
memory. Use `templates/PAPER_TEMPLATE.md`; use papers 06–10 as the depth,
tone, code, and GIF reproducibility standard.

Implement one paper end-to-end at a time: create the directory, README,
CPU-only demonstration, and `scripts_make_gif.py`; generate the GIF; run
the program and `python3 -m pytest scripts/validate_paper.py -k <slug> -v`;
perform a separate accuracy pass against the paper; update the root index
and this plan; then commit and push a conventional `docs: add ...
explainer` checkpoint directly to `master`. Finish with the full validator.
No generated artifact should pretend to reproduce a paper chart unless it
uses the paper’s data; illustrative GIFs must say so.

## Progress

- [x] Paper selection and implementation plan written.
- [x] 11 — SentencePiece.
- [x] 12 — T5.
- [x] 13 — Retrieval-Augmented Generation.
- [ ] 14 — Chain-of-Thought Prompting.
- [ ] 15 — PagedAttention / vLLM.

## Task 1: SentencePiece (`papers/11-sentencepiece/`)

- **Title:** *SentencePiece: A simple and language independent subword
  tokenizer and detokenizer for Neural Text Processing*.
- **Authors/year:** Taku Kudo and John Richardson, 2018.
- **Primary source:** [arXiv:1808.06226](https://arxiv.org/abs/1808.06226).
- **Positioning:** All prior explainers begin after text is already token
  IDs. SentencePiece makes this omitted layer explicit: unlike tools that
  assume pre-tokenized words, it trains subword models directly from raw
  sentences and treats whitespace as an ordinary, reversible symbol. It
  is foundational for multilingual and open-weight LLM pipelines, not a
  language-model architecture.
- **Mechanism requirements:** Explain normalization, the visible
  whitespace marker, subword vocabulary training, and deterministic
  segmentation. Distinguish SentencePiece’s framework from the unigram
  language-model algorithm and BPE algorithms it supports; do not claim
  they are the same objective. Explain reversibility and why pre-splitting
  text is a language-specific assumption.
- **Runnable demonstration:** Implement a tiny, self-contained unigram
  segmentation dynamic program over a fixed toy vocabulary. Assert that
  decoding the chosen pieces exactly restores the normalized input and
  that the selected segmentation has no worse negative log score than a
  competing valid segmentation. Do not require the `sentencepiece` wheel.
- **GIF:** Animate a raw string becoming a visible-whitespace stream and
  then alternative subword segmentations, with the best-path pieces
  winning. Its point is reversibility and vocabulary trade-off, not token
  colors.
- **Practical notes:** Name the `sentencepiece` Python package, Hugging
  Face `tokenizers` / `transformers`, normalization-version locking,
  special-token IDs, byte fallback where relevant, and the operational
  danger of changing tokenizer revisions under a checkpoint.

## Task 2: T5 (`papers/12-t5/`)

- **Title:** *Exploring the Limits of Transfer Learning with a Unified
  Text-to-Text Transformer*.
- **Authors/year:** Colin Raffel, Noam Shazeer, Adam Roberts, Katherine
  Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J.
  Liu, 2019 (revised 2023).
- **Primary source:** [arXiv:1910.10683](https://arxiv.org/abs/1910.10683).
- **Positioning:** Paper 02 explains encoder-only BERT and paper 03
  decoder-only GPT-3. T5 supplies the missing encoder–decoder family and
  turns classification, translation, summarization, and QA into a common
  text-in/text-out interface. Its systematic transfer-learning comparison
  and C4 data context connect directly to the Switch paper’s T5-based
  scaling experiments.
- **Mechanism requirements:** Cover encoder bidirectional attention,
  decoder causal self-attention plus encoder–decoder cross-attention,
  task prefixes, and the span-corruption objective (sentinel tokens and
  target reconstruction). Be careful to distinguish T5’s original
  objective from BERT MLM and later T5 variants.
- **Runnable demonstration:** Build a miniature span-corruption function
  that replaces selected contiguous spans with ordered sentinel IDs and
  emits the target sentinel-plus-removed-span sequence. Assert that the
  sentinels occur in matching order and that reconstruction from input and
  target returns the original token list.
- **GIF:** Show multiple text tasks entering through task prefixes and
  leaving the same decoder, then animate masked spans moving into the
  sentinel-delimited target. The core point is the unified interface.
- **Practical notes:** Name `T5ForConditionalGeneration` and
  `DataCollatorForT5MLM`/T5-compatible span corruption in Hugging Face,
  `t5x` as a forward pointer, decoder label masking, beam-search latency,
  and why instruction strings/templates are part of a model contract.

## Task 3: Retrieval-Augmented Generation (`papers/13-rag/`)

- **Title:** *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks*.
- **Authors/year:** Patrick Lewis et al., 2020 (revised 2021).
- **Primary source:** [arXiv:2005.11401](https://arxiv.org/abs/2005.11401).
- **Positioning:** GPT-3 stores knowledge in weights; RAG joins a
  parametric seq2seq generator to a dense Wikipedia vector index accessed
  through a neural retriever. It is the collection’s first external-memory
  architecture and the conceptual precursor to many production retrieval
  systems. State clearly that RAG retrieval is not a guarantee of factual
  output or citations.
- **Mechanism requirements:** Explain query/document embeddings, maximum
  inner-product retrieval, top-k document marginalization, and the paper’s
  RAG-Sequence versus RAG-Token formulations. Explain which parameters are
  trained/frozen in the paper’s setup only after checking the source.
- **Runnable demonstration:** Use deterministic toy query/document vectors
  and generator likelihoods; retrieve top-k by inner product, compute a
  normalized retriever distribution, and marginalize document-conditioned
  generation probabilities. Assert the marginal is a valid distribution
  and that changing a retrieved document can change the answer ranking.
- **GIF:** Animate a query selecting top-k passages, their retrieval
  weights, and a weighted answer distribution. It must make the
  parametric/non-parametric mixture visible.
- **Practical notes:** Name FAISS, Hugging Face RAG classes, and a vector
  database such as pgvector only as an implementation pointer. Cover chunk
  boundaries, embedding/index version coupling, retrieval latency,
  access-control filtering before retrieval, stale corpus updates, and
  evaluation of retrieval separately from generation.

## Task 4: Chain-of-Thought Prompting (`papers/14-chain-of-thought/`)

- **Title:** *Chain-of-Thought Prompting Elicits Reasoning in Large
  Language Models*.
- **Authors/year:** Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
  Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou, 2022
  (revised 2023).
- **Primary source:** [arXiv:2201.11903](https://arxiv.org/abs/2201.11903).
- **Positioning:** GPT-3 established few-shot prompting; this paper shows
  that demonstrations containing intermediate reasoning can markedly
  improve arithmetic, commonsense, and symbolic tasks in sufficiently
  large LMs. It is an inference-time prompting technique, not training,
  a guarantee of faithful reasoning, or a substitute for verified tools.
- **Mechanism requirements:** Contrast answer-only and rationale-plus-
  answer few-shot exemplars, explain why token generation can decompose a
  hard mapping into intermediate steps, discuss the paper’s scale/emergence
  observation carefully, and distinguish later zero-shot “let’s think step
  by step,” self-consistency, and tool-use variants from the original
  method.
- **Runnable demonstration:** Implement a deterministic toy program that
  enumerates multiple candidate arithmetic reasoning traces, extracts
  answers, and majority-votes valid final answers. Assert that the vote
  picks the known correct answer when a majority of traces are correct;
  label it as an illustration of later self-consistency, not an
  implementation of the original single-chain method.
- **GIF:** Show an answer-only prompt path versus a rationale-token path
  that exposes intermediate state before the final answer; alternatively
  show sampled trace answers converging under a majority vote, with the
  later-method caveat in the caption.
- **Practical notes:** Name Hugging Face generation APIs and structured
  output/verification patterns. Cover token-cost and latency growth,
  prompt injection through demonstrations, reasoning-trace privacy,
  separating an answer from an unverified rationale, and external
  calculator/code execution for high-stakes arithmetic.

## Task 5: PagedAttention (`papers/15-pagedattention-vllm/`)

- **Title:** *Efficient Memory Management for Large Language Model Serving
  with PagedAttention*.
- **Authors/year:** Woosuk Kwon et al., 2023.
- **Primary source:** [arXiv:2309.06180](https://arxiv.org/abs/2309.06180).
- **Positioning:** FlashAttention improves the attention kernel; this
  paper targets the serving system around it. Autoregressive generation
  stores a growing key/value cache whose conventional contiguous allocation
  wastes memory through fragmentation and reservation for unknown output
  lengths. PagedAttention partitions the cache into fixed-size blocks that
  can be non-contiguous physically but logically contiguous for a request.
- **Mechanism requirements:** Explain prefill versus decode, what a KV
  cache contains, logical block tables versus physical GPU blocks,
  allocation/reclamation, and block sharing for parallel sampling or beam
  search. Quantify only values checked in the primary source. Relate—but do
  not conflate—PagedAttention with FlashAttention and OS virtual-memory
  paging.
- **Runnable demonstration:** Implement a CPU block manager for two toy
  sequences with fixed block size and reference counts. Assert logical
  token index to physical block/offset translation is correct, shared
  prefix blocks are not freed until the final reference is released, and
  fragmentation is avoided without contiguous reservations.
- **GIF:** Animate logical tokens growing across non-contiguous physical
  KV blocks, then show a shared prefix being referenced by two requests and
  reclaimed only after both finish. It must communicate why block tables
  save memory.
- **Practical notes:** Name vLLM, Hugging Face Text Generation Inference
  as a contrast, CUDA graph/batching interactions only after source-backed
  qualification, prefix caching, allocator telemetry, block size versus
  internal fragmentation, admission control, and multi-tenant cache
  isolation.

## Completion criteria

For each paper, its individual validator selection and code command pass
before the next begins. After Task 5, run:

```bash
python3 -m pytest scripts/validate_paper.py -v
git diff --check
```

The expected final result is 15 passing paper cases, five new rows marked
`done` in the root README, all Progress boxes checked, one conventional
commit and push per paper, and no unrelated changes.
