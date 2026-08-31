# Efficient Estimation of Word Representations in Vector Space (word2vec)

## 1. TL;DR
Mikolov and colleagues introduced two very small neural objectives, CBOW and
skip-gram, for learning a vector for each word from its neighboring words.
Instead of treating `coffee` and `tea` as unrelated integer IDs, training puts
words used in similar contexts in nearby regions of a vector space. The paper's
point was as much speed as representation quality: removing a costly hidden
layer made it practical to learn useful vectors from very large corpora. These
are static embeddings, so one vector cannot choose a different meaning for
`bank` in a river sentence and a finance sentence.

## 2. Fun Map for First Years
word2vec learns a map where words used in similar neighborhoods stand near each other, like classmates grouped by the clubs they join.

`📝 nearby words → 🧠 adjust word vectors → 🗺️ similar contexts nearby → 🔎 useful comparisons`

Words that appear in similar neighborhoods get similar coordinates. That turns text patterns into numbers that simple programs can compare quickly.

In “I drink coffee” and “I drink tea,” coffee and tea see similar neighboring words. Training moves their vectors so a later program can recognize that similarity.

💻 **CS analogy:** this is an embedding table plus a vector-search score: a word ID looks up a row, and dot products rank candidate neighbors.

### Beginner walkthrough

Read the arrows as a sequence of responsibilities. First identify what enters
the system, then ask what the paper changes, what information is preserved or
discarded, and what leaves the operation. For **skip-gram training with negative sampling**, the key question
is not “does the model sound clever?” but “which intermediate value carries the
new information, and what would go wrong if it were missing?”

### CS student checkpoint

The map corresponds to a small program: input data enters a function, the
paper-specific state or transformation runs, and an assertion checks **positive pairs are rewarded while sampled negatives use the configured frequency distribution**.
The equation `logσ(vᵀv′)+Σlogσ(−vᵀvₙ)` is the compact specification for that function. Trace
one concrete item through each arrow before thinking about larger batches,
parallel hardware, or production optimizations.

## 3. Math Playground
The essential equation or rule is:

```text
p(o|c) = exp(u_oᵀv_c) / Σ_w exp(u_wᵀv_c)
```

**Essential equation:** \(p(o\mid c)=\exp(u_o^Tv_c)/\sum_w\exp(u_w^Tv_c)\). c is a center word and o a nearby word. Their dot product is a “how well do these arrows point together?” score. Exponentials make high scores stand out, and dividing by the sum converts all candidate scores into probabilities that add to 1. Training raises the probability of words that really occur nearby.

The dot product measures how well two word arrows agree. Dividing by the total turns all candidate scores into probabilities that add to 1.

If two vectors point in similar directions, their dot product is large. The exponential makes that large score stand out before the division converts all candidates into a probability distribution.

## 4. Background: What Came Before
Before word2vec, programs often represented a word as a huge one-hot ID or counted co-occurrences in a table. Those representations made related words look unrelated and were costly to use at scale. This paper was needed to make compact, reusable word features practical on very large text collections.

This gave NLP compact features where related words could be discovered from usage instead of manually coded.

This made word meaning usable as a compact numeric feature, though one static vector still cannot separate every sense of a word.

## 5. Why It Matters
Older count tables and one-hot encodings made vocabulary items independent:
an application had to rediscover that `Paris` and `Rome` behave similarly.
Neural language models could learn dense representations, but their hidden
layers and vocabulary-wide output made large-scale training expensive. The
2013 paper isolated the representation-learning step into simple log-linear
models. Its authors reported training high-quality vectors from 1.6 billion
words in under a day, then evaluated relationships such as city/country and
singular/plural using vector offsets.

That changed everyday NLP practice. A frozen embedding matrix became a cheap
input feature for classifiers, taggers, retrieval systems, and sequence models.
Today, transformer token embeddings are trained end-to-end and contextualize a
token with attention, but word2vec remains the clearest introduction to why an
embedding lookup is learnable rather than merely a table of numbers. It also
introduced engineering ideas—sampling, frequency-aware output coding, and
streaming corpus updates—that recur in large-scale training.

## 6. Core Intuition
Imagine learning a map of a city from who regularly shares a table at lunch.
You never receive labels such as “beverage” or “capital”; you only observe
neighbors. If `coffee` often occurs near `tea`, `cup`, and `cafe`, their map
locations should become useful for similar downstream decisions. A word vector
is that location, not a dictionary definition.

```mermaid
flowchart LR
  S["the cat sat on the mat"] --> W["choose a center and window"]
  W --> P["observed center/context pairs"]
  P --> E["adjust embedding vectors"]
  E --> N["similar contexts → similar locations"]
```

CBOW takes surrounding words, averages their vectors, and predicts the middle
word: it is the speedy “fill in the blank” version. Skip-gram reverses the
arrow: one center word predicts each nearby context word. Neither model reads a
whole document or understands a sentence grammar. The useful pressure comes
from many overlapping local windows: words that substitute for one another
receive related training signals.

## 7. The Mechanism
Let each vocabulary item have an input vector \(v_w\) and an output vector
\(u_w\). In skip-gram, a center word \(c\) and one observed neighboring word
\(o\) should score highly through the dot product \(u_o^T v_c\). With a full
softmax, the conditional probability is

\[
p(o\mid c)=\frac{\exp(u_o^T v_c)}{\sum_{w\in V}\exp(u_w^T v_c)}.
\]

Training maximizes the log probability of every center/context pair in a
window. CBOW instead averages or sums its context input vectors and predicts
the center. The original paper used a hierarchical softmax: vocabulary words
are leaves of a Huffman binary tree, so a prediction follows a path of binary
decisions. Frequent words receive short paths, avoiding one score for every
vocabulary entry.

```mermaid
flowchart TD
  C["center: 'with'"] --> V["input vector v_with"]
  V --> O1["score context: coffee"]
  V --> O2["score context: tea"]
  V --> H["hierarchical-softmax path\nor sampled binary objectives"]
  O1 --> U["update input and output vectors"]
  O2 --> U
```

![Illustrative skip-gram pair construction](assets/skipgram_pairs.gif)

The animation is illustrative, not a figure or measured result from the paper.
It shows the later, widely used negative-sampling implementation pattern: for
an observed pair, maximize \(\log\sigma(u_o^T v_c)\), while for a few sampled
noise words \(n\), maximize \(\log\sigma(-u_n^T v_c)\). This turns one huge
multiclass prediction into several binary distinctions. It is important not to
attribute negative sampling to this particular paper's reported architecture;
the paper presents hierarchical softmax, while negative sampling appeared in
the follow-up word2vec paper.

For a positive pair, the derivative increases its dot product. For a negative
pair, it decreases the dot product. Updating both the center and context-side
tables lets context statistics settle into geometry. Cosine similarity is then
often used at query time because it compares direction rather than raw vector
length. The famous analogy calculation, such as `Paris - France + Italy`, is a
nearest-neighbor probe, not a law of language and not evidence that a model has
symbolically reasoned about geography.

The paper compared CBOW and skip-gram on a semantic/syntactic relationship
test. Its 640-dimensional comparison reported stronger semantic accuracy for
skip-gram and stronger syntactic accuracy for CBOW; results depend on corpus,
window, vocabulary, and metric. The central contribution is the efficient
architecture, not a claim that a single benchmark establishes understanding.

### Mechanism in Code

At implementation level, the mechanism operates on center/context ids and sampled negatives. A faithful
forward pass should follow this order: score the positive pair, score negatives, compute binary logistic gradients, and update vectors. Keep the intermediate
representation available while debugging; collapsing everything into one
opaque framework call makes shape and numerical errors much harder to isolate.

The key production failure to guard against is sampling negatives from an unsuitable frequency distribution. Add a tiny
reference test with hand-checkable values, then add a property test that
covers padding, empty/short inputs, boundary probabilities, and the largest
supported shape. Compare intermediate tensors with tolerances appropriate to
the dtype, and log the paper-specific statistic during a canary rollout.


## 8. Practical Engineering Notes
### Worked Math & Dataflow

The compact view below makes the paper's central calculation concrete:

```text
log σ(vᵀv′)+Σ log σ(−vᵀvₙ)
```

In practice, the calculation is a pipeline: Skip-gram increases the score of a real center-context pair and decreases scores for sampled negatives. Negative sampling turns a large vocabulary normalization into a few binary decisions. The important engineering
choice is to preserve the paper's intended invariant while making the operation
fit the available memory, batch size, and evaluation protocol.

```mermaid
flowchart LR
    A[paper input] --> B[center word → positive/negative contexts → vector update]
    B --> C[paper output]
```

![Animated worked-math walkthrough for word2vec](assets/worked_math.gif)


For production, use a maintained implementation rather than this tiny demo.
`gensim.models.Word2Vec` provides skip-gram/CBOW training and vocabulary
management; PyTorch's `nn.Embedding` is the usual building block when the
embedding is part of a larger model. Keep the text normalization, tokenizer,
vocabulary cutoff, and embedding artifact versioned together. A changed
lowercasing rule silently changes IDs and makes an old index or classifier
incompatible.

The output table can dominate memory: two float32 tables of \(|V|\times d\)
need roughly \(8|V|d\) bytes during basic training. Subsampling very common
words, discarding rare words, hierarchical softmax, or negative sampling reduce
compute, but each changes what “co-occurrence” means. Stream corpus shards and
record the random seed and worker count; asynchronous/hogwild-style updates can
be fast but less reproducible. For search, normalize only if the intended score
is cosine similarity, and use an approximate-nearest-neighbor index such as
FAISS when scans no longer fit the latency budget.

Static embeddings encode corpus biases and stale language. Audit nearest
neighbors and downstream slices instead of treating a visually pleasing analogy
as a safety test. They also have no vector for a truly unseen word unless the
pipeline supplies an `UNK` rule; subword methods such as SentencePiece or
fastText address that limitation differently. Do not mix vectors trained with
different preprocessing or dimensions in the same retrieval index.

Evaluate embeddings with the job they will actually serve. Intrinsic neighbor
lists and analogy accuracy are useful regression checks, but they do not prove
that a ranking, moderation, or classification product improves. Pin a held-out
downstream evaluation, watch for vocabulary coverage by language and customer
segment, and decide explicitly how a missing token is represented. If vectors
are exposed through nearest-neighbor search, retain document-level permission
filters outside the vector lookup: embedding proximity is not authorization.

## 9. Runnable Code Example
### Run from the repository root

Prerequisites: Python 3 and the dependencies imported by [`implementations/16-word2vec/code/skipgram_negative_sampling.py`](implementations/16-word2vec/code/skipgram_negative_sampling.py).
The example is intentionally small enough to run on CPU; it is a teaching
implementation, not a production training or serving benchmark.

```bash
python3 implementations/16-word2vec/code/skipgram_negative_sampling.py
```

### What the example demonstrates

Read the module docstring first, then follow the functions implementing
**skip-gram training with negative sampling**. The program turns `logσ(vᵀv′)+Σlogσ(−vᵀvₙ)` into executable operations,
prints a compact result, and checks that **positive pairs are rewarded while sampled negatives use the configured frequency distribution**. The assertion matters:
it tests the semantic contract near the mechanism instead of treating a
plausible final number as proof that the implementation is correct.

### Expected behavior and useful experiments

The command should finish without a traceback and print a successful summary
or assertion message. You should observe the paper-specific behavior, not a
particular random numeric value. Change one input at a time: inspect the
intermediate tensor or state, rerun with a boundary case, and then compare the
result with the expected invariant. A useful first experiment is to **check positive scores against sampled negatives and audit nearest neighbors on held-out relations**.

### Production connection

The toy program does not model every distributed or large-scale concern. In a
real service, version the preprocessing and configuration, record the relevant
intermediate statistic, and measure peak memory, throughput, p95/p99 latency,
and task quality. The first production guard should target **subsampling or negative-sampling bias that produces plausible but unusable vectors**;
preserve a transparent reference path or a canary comparison before replacing
it with a fused, distributed, or highly optimized implementation.

## 10. Common Misconceptions & Pitfalls
- **Misconception: `logσ(vᵀv′)+Σlogσ(−vᵀvₙ)` is the whole implementation.** The equation describes the paper's central relationship, but `skip-gram training with negative sampling` also requires explicit input contracts, ordering, masking or sampling rules, and numerical choices. If those details are left implicit, two implementations can share the same formula and still produce different results. Treat the equation as a contract and document each intermediate tensor or state transition.
- **Misconception: the mechanism is automatically reliable when the final metric looks good.** A model can compensate for a wrong reduction, stale state, or malformed edge/token boundary on common examples. The local guard is **positive pairs are rewarded while sampled negatives use the configured frequency distribution**. Check it on a tiny hand-worked fixture and on adversarial inputs before trusting an aggregate benchmark.
- **Pitfall: optimizing the operation before measuring its actual bottleneck.** For this paper, watch for **subsampling or negative-sampling bias that produces plausible but unusable vectors** rather than assuming the largest theoretical term dominates every workload. Record memory, bandwidth, batch shape, tail latency, and quality slices. An optimization is only safe when it preserves the paper-specific contract and has a rollback path.
- **Pitfall: debugging only the final prediction.** Start with **check positive scores against sampled negatives and audit nearest neighbors on held-out relations**; compare intermediate values with a simple reference. Freeze preprocessing, configuration, seeds, and model versions; then bisect the first divergence. This makes a failure reproducible and distinguishes data-contract errors from numerical instability, integration bugs, and a genuinely unsuitable paper mechanism.

## 11. Quick Concept Checks
**Q:** What is the central idea behind **skip-gram training with negative sampling**?
**A:** It is a structured data or optimization path, not a slogan: inputs are transformed, paper-specific relationships are computed, invalid choices are excluded when necessary, and the result is aggregated into an output or objective. The important implementation question is which intermediate values must remain observable so a reviewer can connect the code to the paper.

**Q:** How should I read `logσ(vᵀv′)+Σlogσ(−vᵀvₙ)`?
**A:** Read each symbol as an operation with a shape, a data source, and a numerical range. Ask what changes when its scale, temperature, rank, timestep, neighborhood, or other paper-specific value changes. Then make a two- or three-example fixture where the expected result can be calculated by hand; this catches notation-to-code misunderstandings early.

**Q:** What invariant must a correct implementation preserve?
**A:** It must preserve **positive pairs are rewarded while sampled negatives use the configured frequency distribution**. This is stronger than asking whether accuracy improved because it is local, deterministic, and testable near the operation that could be wrong. Assert it at the boundary, compare against a small reference implementation, and include the unusual input shape most likely to violate it in production.

**Q:** What is the most dangerous failure mode?
**A:** The first risk to investigate is **subsampling or negative-sampling bias that produces plausible but unusable vectors**. It can produce plausible outputs while degrading only a slice of traffic, so monitor a paper-specific statistic alongside quality and system metrics. A canary should compare the old and new paths on identical inputs and should retain enough intermediate diagnostics to explain a regression.

**Q:** How would I test this idea beyond a happy-path unit test?
**A:** Begin with **check positive scores against sampled negatives and audit nearest neighbors on held-out relations**, then add differential tests against a transparent reference on small randomized inputs. Cover boundaries such as padding, termination, empty neighborhoods, long sequences, rare tokens, extreme values, or duplicated examples when they apply. Test both output values and gradients or state updates when training behavior is part of the paper's claim.

**Q:** What should I remember when applying the paper in a real system?
**A:** Keep the paper's assumptions in the production contract: version the preprocessing and configuration, expose the relevant intermediate statistic, and define quality slices before tuning performance. Compare throughput, peak memory, p95/p99 latency, and task quality against a baseline. The paper is useful only when its mechanism remains correct under the workload and failure modes you actually operate.

## 12. Interview Q&A
**Q:** Walk through **skip-gram training with negative sampling** end to end. How would you implement `logσ(vᵀv′)+Σlogσ(−vᵀvₙ)`?
**A:** Decompose the expression into the actual data path: inputs enter the paper-specific transformation, intermediate scores or states are computed, invalid elements are excluded, and the result is reduced into the output or loss. For this paper, `logσ(vᵀv′)+Σlogσ(−vᵀvₙ)` is an executable contract, not decoration: document tensor shapes, ownership of mutable state, numerical precision, and where batching changes semantics. Keep a small reference implementation beside the optimized path so a reviewer can connect each line of `code` to one term in the equation.

**Follow-up:** What invariant would you assert, and why is it stronger than checking final accuracy?
**A:** Assert that **positive pairs are rewarded while sampled negatives use the configured frequency distribution**. That property is local enough to fail near the defect, whereas accuracy can remain acceptable while a mask, reduction, or state boundary is wrong on a rare input. Add a hand-computed fixture, a randomized differential test against the reference, and shape/dtype assertions at the API boundary. The test should also cover an empty, padded, terminal, high-degree, long-context, or otherwise adversarial case when that input is meaningful for this mechanism.

**Q:** What is the main production trade-off in this paper, and how would you capacity-plan it?
**A:** The central trade-off is that **the mechanism changes both quality behavior and resource use**. Capacity planning therefore needs more than average FLOPs: measure peak memory, memory bandwidth, communication, preprocessing, batch-size sensitivity, and p95/p99 latency on representative distributions. Define a quality budget before optimizing, then compare a simple baseline with the paper mechanism using identical inputs and seeds. A faster path that silently changes tokenization, routing, masking, sampling, or optimization behavior is not an acceptable optimization until its quality impact is measured.

**Follow-up:** Which failure mode would make you roll back first?
**A:** Roll back on evidence of **subsampling or negative-sampling bias that produces plausible but unusable vectors**, especially when the symptom is silent and outputs still look plausible. Add dashboards for the paper-specific statistic, error and timeout rates, resource saturation, and a task metric sliced by difficult inputs. Use a canary or shadow comparison with the previous implementation, retain the old path behind a flag, and make the rollback decision threshold explicit before deployment. The important SDE2 judgment is to protect the paper’s semantic contract, not merely to chase a faster benchmark.

**Q:** A model passes unit tests but fails in production. What is your debugging plan?
**A:** Start with **check positive scores against sampled negatives and audit nearest neighbors on held-out relations**. Reproduce the smallest production-shaped example, freeze the model and preprocessing versions, and compare intermediate tensors or records rather than only the final prediction. Check data contracts, masks, sequence boundaries, random seeds, numerical precision, and serving mode in that order; then bisect between the reference and optimized implementations. If the defect is not numerical, run a controlled ablation that removes the paper-specific mechanism and compare the resulting failure rate, which separates integration problems from a bad mechanism or configuration.

**Follow-up:** What evidence would you present in the review or postmortem?
**A:** Present one minimal failing input, the expected **positive pairs are rewarded while sampled negatives use the configured frequency distribution**, the first intermediate value that diverged, and the regression test that now protects it. Include a before/after table for task quality, memory, throughput, p95/p99 latency, and cost, with slices for the failure population. A complete SDE2 answer also states the rollout guard, owner, and alert threshold. That turns a paper idea into an operable system rather than a one-line claim about an equation.

## 13. Further Reading
- [Original paper](https://arxiv.org/abs/1301.3781)
- [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)
- [Gensim Word2Vec documentation](https://radimrehurek.com/gensim/models/word2vec.html)
- [FAISS documentation](https://faiss.ai/)
