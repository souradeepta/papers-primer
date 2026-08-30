# Learning Transferable Visual Models From Natural Language Supervision (CLIP)

## TL;DR

CLIP learns an image encoder and text encoder together by making matching
image-caption pairs close in one embedding space and mismatched pairs distant.
At inference, candidate labels become text prompts; their text embeddings are
compared with an image embedding to form a zero-shot classifier. The paper
trained on large-scale web image-text pairs and found transfer across many
tasks. It is a matching representation, not a guarantee that a prompt is true.

## Fun Map for First Years 🧭

CLIP puts pictures and captions on one shared map. A photo of a dog should land close to the words “a photo of a dog.”

`🖼️ image + 📝 caption → 🧠 two encoders → 🗺️ shared space → 🔎 compare with prompts`

💻 **CS analogy:** CLIP builds a shared search index: a picture query and a text query should retrieve the same matching record.

## Math Playground 🧮

For a batch, CLIP forms a score table \(S_{ij}=\tau\,I_i^T T_j\): every image vector compares with every text vector. The diagonal pairs are the correct matches, and cross-entropy teaches each row and column to rank its partner highest. It is like testing all query–document pairs in a small search benchmark at once.

## Background: What Came Before 🕰️

Vision models were commonly trained on fixed human-written class labels, so adding a new label set required a new supervised dataset and training run. Text and image systems also tended to live in separate pipelines. CLIP was needed to use plentiful captioned web data to connect the two modalities and make text-defined, zero-shot classification possible.

## Why It Matters

Traditional image classification starts with a fixed label taxonomy and requires
human-curated examples for each class. Natural language supplies a broader form
of supervision: captions, titles, alt text, and nearby prose connect images to
concepts that were never selected as a closed label list. CLIP turns every
minibatch into a retrieval problem: identify the paired caption for an image and
the paired image for a caption. That makes text a flexible output interface.

The paper helped establish contrastive vision-language pretraining as a basis
for zero-shot classification, image search, and later multimodal systems. Its
transfer result does not make web text ground truth. Captions can be noisy,
biased, incomplete, or unrelated to pixels, and benchmark overlap matters. A
newer model called CLIP may use different data, encoders, losses, and safety
policies than the original paper.

## Core Intuition

Imagine placing photographs and their descriptions on the same map. A golden
retriever image should land near “a photo of a golden retriever,” not near “a
photo of a traffic light.” Repeated pairings teach which visual patterns and
phrases occupy neighboring places. To classify a new image, place the candidate
class descriptions on that map and choose the closest one.

```mermaid
flowchart LR
 I[image] --> IE[image encoder]
 T[text caption] --> TE[text encoder]
 IE --> S[normalized similarity matrix]
 TE --> S
 S --> C[contrastive matching loss]
```

The procedure is closer to retrieval than to a classifier with a permanent
softmax vocabulary. Prompts define the candidate classes at inference, which is
useful but sensitive: “a photo of a dog” and “a satellite image of a dog” are
different queries.

## The Mechanism

For a batch of \(N\) paired examples, CLIP encodes images as \(I_i\) and text
as \(T_j\), L2-normalizes both, and forms logits
\(L_{ij}=\tau I_i^TT_j\), with learned temperature \(\tau\). The diagonal
contains observed matches. Cross entropy treats matching text as each image
row's target and matching image as each text column's target; the final loss is
the mean of these directions.

```mermaid
flowchart TD
 B[batch of image-text pairs] --> A[encode and normalize]
 A --> M[N by N similarity logits]
 M --> R[image-to-text cross entropy]
 M --> C[text-to-image cross entropy]
 R --> L[mean contrastive loss]
 C --> L
```

![Illustrative contrastive matching](assets/contrastive_pairs.gif)

At zero-shot classification time, turn each class name into one or more
templates such as “a photo of a {label}.” Encode these prompts, compare a new
image against them, and softmax only over the supplied candidates. The original
paper used prompt ensembling because wording changes results. This is not
open-world detection and it cannot say “none of the above” without an extra
decision policy.

The loss supplies relative information: an image prefers its paired caption to
the other captions in the batch, rather than reconstructing pixels or language.
Batch composition therefore provides negatives, and larger diverse batches give
more comparisons. The GIF is illustrative, not a paper plot. It shows diagonal
matches increasing and mismatched scores decreasing.

The paper used ResNet or Vision Transformer image encoders, a Transformer text
encoder, and 400 million image-text pairs collected from public web sources.
Exact transfer depends on encoder and prompt set. The geometry can capture useful
concepts while absorbing web-data biases, shortcuts, and coverage gaps.

## Practical Engineering Notes

Use a maintained implementation such as OpenCLIP or a package with documented
weights, tokenizer, transform, and license. Resize/crop, RGB handling,
normalization, text context length, and prompts are part of the checkpoint
contract. Embedding offline images with one transform and queries with another
silently changes score distributions.

Normalize vectors if the checkpoint expects cosine similarity, and record
whether an index stores float32, float16, or quantized embeddings. FAISS can
make large retrieval practical, but alters recall and can require reranking.
Keep index, model, prompt, and corpus revisions together. Re-embedding only
some documents after a model upgrade creates incompatible vector spaces.

Evaluate with the exact candidate labels and prompt ensemble used in production.
Synonyms and domain terminology can change rankings. Add a rejection policy or
calibrated downstream head when the product must abstain. A high similarity is
not a probability of truth, identity, safety, or intent. Contrastive models can
rely on shortcut cues and fail under image-text distribution shift.

Image-text data can contain personal information, copyrighted material,
stereotypes, and misleading captions. Apply access control before retrieval,
document data provenance, and red-team harmful associations. People-related
uses need task-specific privacy and fairness review, not extrapolation from a
broad benchmark. Rate limit embedding APIs and establish a policy for sensitive
similarity search.

### Evaluation and operations

Treat prompts as versioned configuration. A broad concept can need several
templates and class-name variants; average normalized text embeddings only after
validating on a held-out set. Keep the exact prompt set with the deployment so
a copy edit cannot silently change rankings. For retrieval, test both recall and
the relevance of returned content, not only a nearest-neighbor distance.

Log transform revision, tokenizer failures, input dimensions, score margins,
candidate-set size, rejection rate, and latency under an approved telemetry
policy. Sudden confidence changes can reveal a preprocessing bug, not improved
semantics. Review representative failures, including rare classes and new
domains, with a documented evaluation rubric.

ANN retrieval needs the same access controls as keyword search: similarity does
not grant access to a private image or document. Build and validate a new index
revision before switching traffic, retain the previous revision for rollback,
and compare a canary sample against exact search. Production quality depends on
this surrounding system as much as on the contrastive loss.

Finally, useful zero-shot ranking is not a safe automated decision. High-stakes
uses need domain-specific data, error analysis, abstention or human review where
appropriate, and post-release monitoring. Broad web supervision is valuable for
transfer but does not establish fairness, completeness, or suitability for a
particular population.

### Integration checklist

Before release, run a deterministic embedding canary: a small approved set of
images and prompts with expected nearest-neighbor order and score tolerances.
It catches swapped color channels, changed interpolation, tokenizer upgrades,
or accidental model substitutions. Include an end-to-end test that builds an
index, executes an authorized query, applies filtering, and verifies that a
denied record is never returned. This checks a product property that contrastive
training itself cannot express.

Use separate evaluation sets for prompt selection and final reporting. Repeated
prompt edits against one benchmark can overfit wording to that benchmark just as
repeated hyperparameter tuning can overfit a validation set. For multilingual
or specialized domains, evaluate the actual language and terminology rather
than assuming English web supervision transfers. An absent label, unfamiliar
script, or specialized image modality should be represented as uncertainty,
not silently forced into the nearest broad class.

Embedding caches need lifecycle management. Associate each vector with model,
transform, tokenizer, source, and permission revisions; delete or rebuild it
when retention policy requires. If a document is removed, propagate removal to
the vector index and its backups. These details make CLIP's flexible matching
capability reliable within a real application rather than only in a notebook.

## Runnable Code Example

[`code/contrastive_ranking.py`](code/contrastive_ranking.py) checks a toy score
matrix in image-to-text and text-to-image directions.

```bash
python3 papers/20-clip/code/contrastive_ranking.py
```

It isolates the ranking invariant; real CLIP computes encoder vectors, a
temperature-scaled matrix, and cross-entropy gradients over a batch.

## Common Misconceptions & Pitfalls

**“Zero-shot means no choices are needed.”** Candidate labels and prompt wording
can materially change accuracy and bias.

**“Similarity is factual confidence.”** It is a relative compatibility score,
not a calibrated claim about reality.

**“CLIP detects every object.”** Standard zero-shot scoring compares a whole
image to supplied text; detection requires additional methods.

## Interview Q&A

**Q:** What is CLIP's training target?
**A:** It identifies matching image-text pairs within a batch in both directions.

**Q:** Why normalize embeddings?
**A:** It makes dot products behave as cosine similarities and stabilizes the
comparison geometry.

**Q:** How does zero-shot classification work?
**A:** Encode prompted labels and choose the text embedding most similar to the
image embedding.

**Q:** Why is temperature learned?
**A:** It controls the sharpness of contrastive softmax logits.

**Q:** What is a deployment risk?
**A:** A preprocessing, tokenizer, index, or prompt mismatch can invalidate
scores without a runtime error.

## Further Reading

- [Original paper](https://arxiv.org/abs/2103.00020)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [FAISS documentation](https://faiss.ai/)
