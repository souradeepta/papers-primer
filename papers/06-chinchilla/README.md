# Training Compute-Optimal Large Language Models (Chinchilla)

Hoffmann et al., 2022 — [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)

## TL;DR

Chinchilla asks a planning question rather than proposing a new Transformer block: with a fixed pre-training compute budget, how should we split it between model parameters and training tokens? The paper finds that many prominent language models had spent too much of their budget on parameter count and too little on data. Across more than 400 training runs, its three estimation methods all said that compute-optimal parameter count and token count should grow at roughly the same rate as compute grows. The headline validation was Chinchilla: a 70B-parameter model trained on 1.4T tokens, using roughly Gopher's training compute but outperforming the 280B-parameter Gopher on the reported downstream suite. This is a scaling-law result: it is a useful empirical rule for the experimental regime, not a law that makes data quality, architecture, hardware, or post-training irrelevant.

## Why It Matters

Before this paper, a very visible story about language-model progress was “make the dense model larger.” That story had real evidence behind it: the earlier Kaplan scaling-law work showed smooth power-law improvements with model size, data, and compute. But a project does not get to increase all three independently. A team normally knows its accelerator budget and deadline first. At that fixed budget, making each training step more expensive by increasing the number of parameters means it can afford fewer steps or fewer tokens. Conversely, training an extremely small model for a huge number of tokens leaves representational capacity unused. The engineering decision is an allocation problem.

The distinction matters because a parameter is paid for more than once. It costs compute during pre-training, but it also occupies memory, increases the cost and latency of every generated token, constrains batch size and serving hardware, and can make fine-tuning less convenient. A model that is smaller yet better trained can be cheaper over its entire useful life. The paper explicitly frames this lifecycle benefit: training cost is amortized by later inference and fine-tuning.

The contemporary comparison makes the point concrete. The paper lists GPT-3 at 175B parameters and 300B training tokens, Gopher at 280B and 300B tokens, and Chinchilla at 70B and 1.4T tokens. This does not mean “70B is universally ideal.” It means that, for approximately the compute used for Gopher, the authors' fitted frontier expected a model about four times smaller trained on about four times more data to be a better choice. They then tested that prediction rather than stopping at a curve fit.

Chinchilla also shifted attention to the data supply. If keeping a model on the efficient frontier requires many more tokens as compute rises, progress becomes constrained by collecting, cleaning, licensing, deduplicating, mixing, and documenting data. More tokens are not automatically more useful tokens; repeated low-quality data or leakage into benchmarks can invalidate the intended comparison. The result therefore made data engineering a first-class scaling concern alongside distributed-training engineering.

There is a subtle historical correction here. Kaplan et al. had concluded that, at fixed compute, model size should grow faster than the number of training tokens. Chinchilla obtained approximately equal growth. Hoffmann et al. identify methodological differences, including varying the learning-rate schedule with training horizon and using larger models in their analysis. One should not caricature the earlier work as “wrong”: both studies fit empirical scaling relationships under particular choices and extrapolate beyond the smallest runs. The useful lesson is to validate planning laws with controlled sweeps that resemble the regime where they will be used.

## Core Intuition

Imagine opening a restaurant with a fixed payroll. You can hire a very large kitchen but only afford to serve a few customers, or hire a tiny kitchen and spend all day serving more customers than it can learn to handle. Neither extreme is good preparation for a busy service. A training budget has the same tension: parameters are the kitchen's capacity; tokens are its experience with orders.

At fixed compute, the two knobs are coupled. A larger dense Transformer processes a token with more multiply-adds. So, if the total multiply-add budget stays fixed, increasing parameters reduces the affordable token count. The final language-model loss usually has two visible sources of avoidable error: insufficient capacity and insufficient data/optimization. Adding parameters lowers the first; seeing more tokens lowers the second. The efficient choice is where the marginal payoff from either direction is balanced.

```mermaid
flowchart LR
    C[Fixed accelerator budget] --> N[Choose parameters N]
    C --> D[Choose training tokens D]
    N -. "larger N costs more/token" .-> D
    N --> A[capacity-limited error falls]
    D --> B[data-limited error falls]
    A --> L[final pre-training loss]
    B --> L
    L --> F[choose the low point on the iso-compute curve]
```

This is why a loss-versus-model-size plot at one fixed compute budget has a valley. On the left, the model is small and gets plenty of tokens but lacks capacity. On the right, the model is huge but receives too little training. The low point is not necessarily a single sharp integer; it is a useful region, and uncertainty in the fit, data mix, and system efficiency should be part of any real decision.

“Scale parameters and tokens equally” is easy to misread as “keep tokens-per-parameter constant under every definition.” The paper's claim is about how the *optimal* values scale as the available training compute (C) changes: (N_{opt}\) and (D_{opt}\) each have an exponent near one half. Because dense training compute is approximately proportional to (N D), doubling both quantities approximately quadruples compute. The proportionality constant and the resulting tokens-per-parameter ratio come from the empirical fit and model/training setup; they are not supplied by the exponent alone.

## The Mechanism

The authors formalize the decision as minimizing final pre-training loss (L(N,D)), where (N) is non-embedding parameter count and (D) is the number of training tokens, subject to a compute constraint. Their approximate dense-Transformer accounting is (C \approx 6ND) FLOPs. The constant is a planning approximation; actual cost also depends on sequence length, attention, vocabulary/embeddings, activation recomputation, hardware utilization, and implementation details. Still, it captures the central product constraint: at fixed (C), (D \approx C/(6N)).

Their third approach fits a parametric loss surface:

\[
\hat L(N,D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}.
\]

Here (E) is an irreducible asymptote for the data distribution, the (N) term represents finite-model error, and the (D) term represents the penalty for finite training data/tokens. This is not a derivation from first principles. It is a compact empirical model whose quality must be assessed from held-out runs. Under (C\approx6ND), minimizing this surface yields power-law optimal allocations. The exponents are (a=\beta/(\alpha+\beta)) and (b=\alpha/(\alpha+\beta)) for (N_{opt}\propto C^a) and (D_{opt}\propto C^b).

```mermaid
flowchart TD
    R[Many small-to-medium training runs] --> X[Record N, D, final smoothed loss]
    X --> M1[Approach 1: envelope of training curves]
    X --> M2[Approach 2: isoFLOP profiles]
    X --> M3[Approach 3: fit L-hat(N,D)]
    M1 --> P[fit Nopt(C), Dopt(C)]
    M2 --> P
    M3 --> P
    P --> V[Predict a Gopher-compute model]
    V --> CH[Train Chinchilla: 70B parameters, 1.4T tokens]
    CH --> E[Evaluate against Gopher and other reported baselines]
```

The paper intentionally uses three estimation routes to reduce dependence on one fitting choice. Approach 1 holds model sizes fixed, trains each over several horizons, interpolates smoothed curves, and takes the lowest-loss envelope at each FLOP count. It reports exponents 0.50 for parameters and 0.50 for tokens. Approach 2 constructs fixed-compute (IsoFLOP) profiles: for each of nine FLOP budgets, it varies model size, determines the compatible token count, fits the loss valley, then fits a power law through the minima. It reports 0.49 and 0.51. Approach 3 fits the parametric surface above using Huber loss and L-BFGS and reports 0.46 and 0.54. The agreement is approximate rather than exact, which is the appropriate reading of an empirical frontier.

The experiments cover more than 400 language models, from 70M to over 16B parameters, trained from 5B to 500B tokens according to the abstract. The extrapolation target was much larger, so the crucial test was the final model. Chinchilla used 70B parameters and 1.4T tokens at approximately Gopher's compute budget; the paper reports 67.5% average MMLU accuracy, more than seven percentage points above Gopher. It also reports broad downstream advantages. Those are evaluation claims under the paper's protocols, not a guarantee that every downstream application will see the same margin.

The animation below uses a deliberately simple symmetric teaching loss under fixed normalized compute. It visualizes the valley, not the authors' measured points or fitted coefficients.

![Illustrative fixed-compute loss valley](assets/fixed_compute_valley.gif)

One operational detail deserves emphasis: the learning-rate schedule must be part of a fair token-budget experiment. The paper argues that decaying the learning rate over a horizon matched to the training-token horizon gives the best final loss in its setup. Reusing a long schedule and reading intermediate checkpoints can make short runs look worse than properly scheduled short runs. In other words, a scaling sweep is an experiment about a *training recipe*, not only a model shape.

## Practical Engineering Notes

Treat a scaling law as a budget-planning prior, then calibrate it with runs in your own regime. Tokenizers change token counts; code, multilingual text, synthetic data, and repeated epochs differ in information content; architecture changes can alter scaling; and loss may not track the product metric your product cares about. Start with a logarithmic grid of model sizes and token horizons, keep a data-mixture version fixed, and record both realized FLOPs and wall-clock throughput. The `6ND` figure is useful for comparing plans, but GPU-hour cost should include attention overhead, communication, checkpointing, evaluation, failures, and utilization.

Data accounting needs to be explicit. Log unique source tokens, tokens presented to the optimizer, repeat rate, filters, deduplication method, and contamination checks. “1T tokens” without a tokenizer, mixture, and repeat policy is not a reproducible compute/data budget. If high-quality data is the bottleneck, blindly repeating it may change the overfitting and memorization behavior that the clean infinite-data approximation abstracts away.

For production, smaller compute-optimal models can improve latency, memory headroom, and fine-tuning cost, but serving is not determined by parameter count alone. Context length, KV-cache size, quantization, batch shape, speculative decoding, and memory bandwidth often dominate. Measure with the intended workload in frameworks such as PyTorch, JAX, or TensorFlow and serving systems such as vLLM, TensorRT-LLM, or Hugging Face Text Generation Inference. A model selected for pre-training FLOP efficiency can still be the wrong latency/cost point for a particular deployment.

When running a sweep, use a shared configuration schema and immutable run metadata: parameter-count convention, number of layers/width/heads, sequence length, optimizer, batch schedule, learning-rate schedule, data snapshot, seed, and compute estimator. Compare loss after matching token or compute budgets as appropriate; avoid quietly comparing an undertrained large checkpoint with a converged small one. Track confidence intervals across seeds if the differences near the valley are small.

Finally, separate pre-training planning from post-training. Instruction tuning, preference optimization, retrieval, tool use, and architecture changes may move product quality substantially without following the same frontier. Chinchilla answers a valuable narrower question: how to allocate dense autoregressive pre-training compute between parameters and tokens in the studied setting.

## Runnable Code Example

[`code/compute_optimal_scaling.py`](code/compute_optimal_scaling.py) is a CPU-only, dependency-free toy sweep. It holds normalized compute (N D) fixed, evaluates a simplified version of the paper's loss shape for candidate parameter counts, and derives the token count from the fixed budget. The printout marks the lowest point and assertions check that the balanced allocation beats both extremes.

Run it from this directory with:

```bash
python3 code/compute_optimal_scaling.py
```

The constants are intentionally illustrative. The point is executable intuition: if loss has both a capacity-limited and data-limited contribution, a product constraint creates an interior optimum. It does not fit the Chinchilla data, train a language model, or estimate a real hardware budget.

## Common Misconceptions & Pitfalls

**“Chinchilla says every model needs exactly 20 tokens per parameter.”** The paper's central result is near-equal *scaling exponents* for optimal parameters and tokens as compute increases. A convenient rule of thumb derived from a particular fit is not a universal invariant across model families, datasets, or definitions of parameters/tokens.

**“Smaller always beats larger.”** No. For a fixed budget, the paper predicts a specific trade-off. If the budget rises, both the optimal model and optimal data budget rise. A model can also be too small and capacity-limited.

**“More tokens means more distinct high-quality data.”** Not necessarily. A token count can include repeated epochs, low-quality additions, or synthetic data. The paper's interpretation assumes a sufficiently large data regime; data quality and contamination are separate engineering and scientific concerns.

**“The 6ND FLOP formula is an exact bill.”** It is an approximation useful for a dense-Transformer scaling analysis. Sequence length, attention cost, embeddings, communication, and accelerator utilization can make wall-clock and monetary costs differ materially.

**“A lower pre-training loss guarantees better behavior.”** Loss is an important proxy and Chinchilla reports downstream evaluations, but safety, factuality, instruction following, domain fit, and latency require their own measurements and post-training work.

## Interview Q&A

**Q:** What question does Chinchilla answer?
**A:** Given fixed dense language-model training compute, it asks how to allocate that budget between parameter count and training-token count to minimize final pre-training loss.

**Q:** Why is there an optimum instead of simply maximizing parameters?
**A:** Approximate training compute grows with the product of parameters and tokens. At fixed compute, more parameters leave fewer tokens, so gains in capacity eventually lose to undertraining.

**Q:** What is an IsoFLOP profile?
**A:** It is a set of runs at a fixed FLOP budget with different model sizes and therefore different compatible token counts. Plotting final loss against model size reveals a valley whose minimum estimates the efficient allocation for that budget.

**Q:** State the parametric loss model used in the paper.
**A:** The fitted form is \(\hat L(N,D)=E+A/N^\alpha+B/D^\beta\): an irreducible term plus decreasing finite-model and finite-data terms.

**Q:** What empirical scaling result did the three approaches agree on?
**A:** They found parameter count and training tokens should each scale roughly as the square root of compute, so both should approximately double when the compute budget quadruples.

**Q:** How did the paper validate an extrapolated scaling prediction?
**A:** It trained Chinchilla, a 70B-parameter model on 1.4T tokens at about Gopher's compute budget, then compared it with the reported large-model baselines on downstream evaluations.

**Q:** Why can a compute-optimal smaller model be operationally attractive?
**A:** It can deliver stronger quality at a given training budget while lowering later fine-tuning and inference cost relative to a much larger, less fully trained model. Serving measurements are still necessary.

## Further Reading

- [Original paper: Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [Chinchilla paper HTML, including methods and appendices](https://arxiv.org/html/2203.15556)
- [Scaling Laws for Neural Language Models (Kaplan et al.)](https://arxiv.org/abs/2001.08361)
- [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027)
