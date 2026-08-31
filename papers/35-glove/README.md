# GloVe: Global Vectors for Word Representation

**Pennington, Socher & Manning, 2014** · EMNLP 2014

## TL;DR

GloVe learns word vectors from global word co-occurrence counts. It fits dot
products between word and context vectors to the logarithm of observed counts,
with a weighting function that prevents very rare or very frequent pairs from
dominating. The result captures useful semantic and syntactic relationships in
a compact vector space.

## Fun Map for First Years 🧭

text corpus 📚 → nearby-word counts 🔢 → weighted statistics ⚖️ → vectors 📍 → similarity and analogies 🔗

If “ice” often appears near “cold” and “steam” near “hot,” their count
patterns contain meaning even without labeled definitions. GloVe turns these
shared neighborhood patterns into positions in vector space.

💻 **CS analogy:** A co-occurrence matrix is like a sparse service-usage table;
matrix factorization compresses recurring usage patterns into small feature
vectors.

## Math Playground 🧮

The central goal is w_i dot w_tilde_j plus b_i plus b_tilde_j approximately
equals log of X_ij.

```text
J = Σ_(i,j) f(X_ij) [w_iᵀ w̃_j + b_i + b̃_j − log(X_ij)]²
```

X_ij is the number of times word i occurs near context word j. A dot product
acts as a compatibility score. Taking a logarithm makes huge raw count
differences easier to fit, while GloVe weights each pair to limit extremes.

## Background: What Came Before 🕰️

Count-based distributional methods used large co-occurrence matrices, while
predictive methods such as word2vec learned embeddings by predicting local
contexts. Both observed that nearby-word statistics encode useful meaning.

GloVe combined global counts with a vector-learning objective. It gave a
simple way to factorize corpus-wide statistics while retaining small dense
embeddings useful in downstream models.

## Why It Matters

GloVe became a standard pretrained embedding source and a clear demonstration
of how global corpus structure can become vectors. It complements paper 16,
word2vec, which emphasizes local predictive training.

## Core Intuition

Words are represented by how their context distributions compare, not by their
spelling. The important signal is often a ratio: a context that is common near
“ice” but not “steam” helps distinguish the two.

## The Mechanism

The model keeps separate word and context embeddings plus biases. For each
nonzero co-occurrence pair it minimizes a weighted squared error between its
score and the log count. After training, word and context representations are
often combined for downstream use.

![Animation of co-occurrence structure becoming vector geometry](assets/counts-to-vectors.gif)

```mermaid
flowchart LR
    C[corpus] --> X[sparse co-occurrence counts X_ij]
    X --> W[weighted log-count objective]
    W --> V[word and context vectors]
    V --> U[similarity or downstream model]
```

## Practical Engineering Notes

Build sparse co-occurrence tables with a defined window and distance weighting.
Avoid materializing a full vocabulary-square matrix. Pretrained embeddings
need vocabulary, casing, and licensing checks; static embeddings cannot resolve
different senses of the same word from sentence context.

## Runnable Code Example

Run python3 implementations/35-glove/code/glove_weighted_least_squares.py.
It first builds a sparse, distance-weighted co-occurrence table from a small
tokenized corpus, then optimizes word/context vectors and biases over observed
pairs only. The final assertion confirms that weighted log-count loss falls
during training.

## Common Misconceptions & Pitfalls

- GloVe is not a contextual embedding model; each word type has one vector.
- A high co-occurrence count alone is not semantic proof; corpus bias matters.

## Interview Q&A

**Q:** What data does GloVe learn from?  
**A:** Word-context co-occurrence counts from a corpus.

**Q:** Why use log counts?  
**A:** They compress the wide numeric range of raw frequency.

**Q:** Why have context vectors too?  
**A:** They model asymmetric word-context roles during factorization.

**Q:** How does GloVe differ from word2vec?  
**A:** GloVe explicitly fits global counts; word2vec uses a predictive objective.

**Q:** What is a major limitation?  
**A:** One static vector cannot choose a different sense per sentence.

## Deeper Mechanism and Engineering

Build a co-occurrence table by scanning a corpus with a context window. Each
time word i occurs near word j, increment X_ij, often by a distance-dependent
weight so closer neighbors count more. The table is sparse: most vocabulary
pairs never occur together, and efficient implementations iterate only over
observed pairs.

GloVe gives every vocabulary item a word vector, a context vector, and two
biases. Its predicted log count is the dot product of the two vectors plus
their biases. A weighted squared error compares that prediction with log X_ij.
The weighting function suppresses extremely rare noisy pairs and prevents a
handful of very frequent pairs from dominating optimization.

Why retain separate word and context embeddings? A word in the center and a
word in the surrounding window have related but not identical roles. The
factorization uses that asymmetry while learning. After training, applications
often add the two embeddings or choose the word embedding, but that downstream
choice should be evaluated rather than assumed.

Global statistics can reveal relationships that a local objective sees only
through individual windows. If two words have similar ratios of co-occurrence
with many contexts, their vectors can become similar even when the words rarely
appear directly beside one another. The approach remains limited by corpus
coverage, tokenization, and social bias encoded in text frequency.

In production, precompute counts with a stable tokenizer and record window
size, vocabulary cutoff, distance weighting, and corpus version. A dense
vocabulary-square array becomes impractical quickly; use sparse maps, shards,
or streaming aggregation. Validate that unknown tokens, case folding, and
punctuation handling match the downstream system.

Static embeddings are cheap and useful for small models, similarity search, or
interpretable baselines. They cannot choose different vectors for “bank” in a
river sentence versus a finance sentence. Contextual encoders solve that by
computing a representation from the full input, but they are more expensive
and do not remove corpus bias automatically.

The objective has a friendly interpretation. For every observed pair, the
model makes a prediction for the logarithm of its count. If “ice” and “cold”
occur together frequently, their word-context dot product and biases should
make a larger prediction than for a rare pair. Squaring the difference makes
large mistakes costly; the weighting function decides how much each observed
pair should influence the total lesson.

The weighting function is important because corpus counts have a long tail.
Pairs seen once may be noise, while punctuation or function-word pairs may be
so common that they crowd out everything else. GloVe increases influence up to
a chosen count threshold and then caps it. This is like a monitoring system
that listens more to repeated signals than one-off events but refuses to let a
single noisy metric dominate every dashboard decision.

Window construction determines what “context” means. A symmetric window sees
left and right neighbors; a directional window can preserve order. Weighting
nearby tokens more strongly encodes a belief that close words carry more
direct relations. These choices, vocabulary cutoff, casing, and tokenization
are part of the learned representation, so they belong in experiment metadata
and reproducible data pipelines.

The final vectors support cosine similarity because vector direction captures
a usage pattern better than raw magnitude alone. Yet a nearest-neighbor list
is an audit tool, not a certificate of meaning. Inspect for stereotypes,
unexpected tokenization fragments, and overly frequent words. For a product,
evaluate retrieval or downstream task behavior on the people and language
varieties that actually matter, then consider contextual models when one static
vector per word is too coarse.

The paper motivates ratios of co-occurrence probabilities because they make
comparisons sharper. A context word such as “solid” may be much more likely
near “ice” than “steam,” while “gas” reverses that contrast. A raw count does
not tell the whole story because common context words appear beside almost
everything. Relative patterns are what let an embedding separate related but
different concepts.

Training samples only nonzero entries of the sparse table. That is a major
engineering saving: a vocabulary of 100,000 words has ten billion possible
pairs, but most do not occur in a realistic window. Store integer ids and
counts, batch observed pairs, and monitor the weighted objective separately
from downstream quality. A lower reconstruction loss is useful evidence, not
a guarantee that the vectors help a classifier or retrieval service.

GloVe sits in an instructive historical middle ground. It has an explicit
statistical object—the corpus count matrix—like classical distributional
semantics, but it learns dense vectors with gradient descent like neural
methods. That clarity is why it remains a good baseline and teaching tool.
Modern contextual models add flexibility, yet global-count embeddings can be
smaller, easier to inspect, and appropriate when compute or data is limited.

## Further Reading

- [Original paper](https://aclanthology.org/D14-1162/)
- [word2vec](https://arxiv.org/abs/1301.3781)
- [SentencePiece](https://arxiv.org/abs/1808.06226)
