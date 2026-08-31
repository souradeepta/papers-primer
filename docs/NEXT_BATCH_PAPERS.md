# Completed Batch: Foundational Sequence Learning

| # | Paper | Primary source | Collection gap |
|---|---|---|---|
| 31 | Long Short-Term Memory | Hochreiter & Schmidhuber (1997) | gated recurrent memory before Transformers |
| 32 | Sequence to Sequence Learning with Neural Networks | [arXiv:1409.3215](https://arxiv.org/abs/1409.3215) | encoder-decoder learning for variable-length sequences |
| 33 | Neural Machine Translation by Jointly Learning to Align and Translate | [arXiv:1409.0473](https://arxiv.org/abs/1409.0473) | soft alignment, the direct predecessor of Transformer attention |
| 34 | Dropout | [JMLR 15(56), 2014](https://www.jmlr.org/papers/v15/srivastava14a.html) | regularization used throughout modern deep learning |
| 35 | GloVe: Global Vectors for Word Representation | Pennington, Socher & Manning (2014) | global-count word embeddings alongside word2vec |

Implemented in dependency-aware order: 31, 32, 33, 34, 35. Every paper now
follows SPEC.md, links one top-level implementation, cites its primary source,
and includes a purpose-built explanatory GIF.
