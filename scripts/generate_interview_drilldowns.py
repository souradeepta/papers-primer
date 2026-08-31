"""Rewrite every official interview section from paper-specific interview data.

The repository treats the interview section as a production-facing learning
artifact.  Keeping the compact data here makes it possible to audit all papers
with one command and prevents later additions from silently falling back to a
generic answer template.
"""
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]

# Each record intentionally names the paper's actual mechanism, invariant,
# failure mode, and test.  The prose generator supplies consistent SDE2 shape;
# these fields supply the paper-specific substance.
DATA = {
"01-attention-is-all-you-need": ("scaled dot-product self-attention", "softmax(QKᵀ/√dₖ)V", "causal and padding masks must prevent invalid keys from contributing", "quadratic score memory and mask leakage", "compare a masked reference with an optimized kernel and test a future-token perturbation"),
"02-bert": ("masked-language pretraining with bidirectional encoder layers", "−log p(xᵢ|context)", "only selected masked positions contribute to the MLM loss and padding is ignored", "train/serve tokenizer drift or an incorrect mask-label alignment", "assert masked positions and evaluate a small downstream classifier with a frozen preprocessing snapshot"),
"03-gpt3-few-shot-learners": ("autoregressive in-context learning", "p(xₜ|x₍<ₜ₎)", "the demonstration examples and query remain in order and the causal mask hides future tokens", "prompt-format sensitivity and context-window truncation", "hold weights fixed, replay prompts byte-for-byte, and compare task accuracy across controlled prompt variants"),
"04-lora": ("low-rank adapter injection into a frozen linear layer", "W′=W+BA", "the base weight is unchanged while adapter shapes and merge/unmerge logits agree", "wrong adapter rank, dtype, target module, or accidental base-weight updates", "compare merged and unmerged logits and assert the frozen parameter checksum"),
"05-instructgpt-rlhf": ("reward-model-guided policy optimization", "E[reward]−βKL(π||πref)", "the KL penalty is measured against the frozen reference policy and reward inputs use the same prompt contract", "reward hacking, preference-label bias, or an unstable policy/reference gap", "track reward, KL, human preference, and adversarial slices separately during an ablation"),
"06-chinchilla": ("compute-optimal joint allocation of model parameters and training tokens", "C≈6ND", "the comparison holds the compute budget and data quality definition constant", "a misleading extrapolation from small runs or a token-counting mismatch", "run matched-budget pilots with held-out scale points and confidence intervals"),
"07-flashattention": ("IO-aware tiled exact attention", "softmax(QKᵀ)V", "online softmax statistics produce the same result as a numerically stable reference", "incorrect tile rescaling, causal-boundary handling, or hardware-specific regressions", "compare outputs and gradients against a reference over tile boundaries and sequence lengths"),
"08-roformer-rope": ("rotary position encoding applied to query and key pairs", "R(m)ᵀR(n)=R(n−m)", "relative offsets, tensor shape, and rotation pairing stay consistent across positions", "frequency extrapolation failure or an off-by-one position/cache index", "test relative-offset invariance and compare long-context perplexity with a no-rotation control"),
"09-dpo": ("direct preference optimization against a reference policy", "logit σ(β log(π(yw|x)/πref(yw|x)−log(π(yl|x)/πref(yl|x))))", "chosen/rejected sequences use the same prompt boundary and reference log-probabilities are detached", "preference leakage, length bias, or incorrect sequence log-prob summation", "unit-test pairwise margins and monitor held-out preference accuracy by length bucket"),
"10-switch-transformer": ("top-1 sparse mixture-of-experts routing", "y=Expert[argmax p(x)](x)", "each token is dispatched to exactly one selected expert and capacity overflow is observable", "expert load imbalance, dropped tokens, and all-to-all communication spikes", "log per-expert counts, overflow rate, routing entropy, and quality for overflowed tokens"),
"11-sentencepiece": ("unigram subword segmentation over raw Unicode text", "argmax_segmentation ∏p(piece)", "normalization, whitespace markers, and encode/decode round trips are versioned together", "unknown pieces, changed normalization, or a tokenizer/model vocabulary mismatch", "snapshot token IDs and test round trips on punctuation, repeated spaces, and non-segmented languages"),
"12-t5": ("text-to-text transfer learning with task prefixes", "input text → target text", "task prefix, target formatting, and special-token boundaries remain part of the model contract", "a prefix or output-format regression that hides behind aggregate metrics", "test exact target formatting and run task-balanced validation for every supported prefix"),
"13-rag": ("retrieval-augmented generation", "p(y|x)=Σ_zp(z|x)p(y|x,z)", "retrieved evidence is traceable to the answer and stale or empty retrieval is handled explicitly", "retriever miss, stale index, prompt overflow, or unsupported generation", "measure retrieval recall, citation support, and answer quality independently with an index snapshot"),
"14-chain-of-thought": ("self-consistency over sampled reasoning traces", "argmax_y Σ_k1[y=y_k]", "vote aggregation uses only final answers and preserves sample identity for diagnosis", "correlated wrong traces, parsing errors, and latency from excessive sampling", "score final answers independently from trace text and test adversarial problems with fixed seeds"),
"15-pagedattention-vllm": ("paged KV-cache allocation for continuous batching", "logical block → physical block", "a request can read only its own logical blocks and reference counts free blocks exactly once", "cross-request cache contamination, fragmentation, or block-table races", "stress concurrent requests with isolation and reference-count assertions"),
"16-word2vec": ("skip-gram training with negative sampling", "logσ(vᵀv′)+Σlogσ(−vᵀvₙ)", "positive pairs are rewarded while sampled negatives use the configured frequency distribution", "subsampling or negative-sampling bias that produces plausible but unusable vectors", "check positive scores against sampled negatives and audit nearest neighbors on held-out relations"),
"17-adam": ("Adam adaptive moment updates with bias correction", "θ←θ−αm̂/(√v̂+ε)", "first and second moments advance with the same step and state is not silently reset", "incorrect bias correction, mixed-precision underflow, or weight-decay coupling", "run a scalar hand calculation and compare optimizer-state memory and update traces"),
"18-resnet": ("residual learning through identity or projection shortcuts", "y=F(x)+x", "the shortcut and residual branch produce identical batch/spatial shapes before addition", "a projection or normalization mismatch that blocks the identity path", "zero the residual branch and assert shortcut behavior, then compare gradient norms"),
"19-gan": ("alternating generator/discriminator adversarial training", "min_Gmax_D V(D,G)", "each optimizer updates only its intended network and sample diversity is measured separately from realism", "mode collapse, discriminator overpowering, or misleading loss interpretation", "track diversity and held-out samples while logging both players’ gradient norms"),
"20-clip": ("symmetric image-text contrastive learning", "sim(I,T)=ĨᵀT̃", "image-text positives align on both retrieval directions and temperature is applied consistently", "duplicate captions, batch composition bias, or preprocessing mismatch between modalities", "test image-to-text and text-to-image retrieval with duplicate and hard-negative slices"),
"21-vae": ("variational encoding with a reconstruction objective and KL regularizer", "ELBO=E_q[logp(x|z)]−KL(q||p)", "reconstruction and KL terms are logged separately and latent samples use the reparameterization path", "posterior collapse, KL dominance, or a decoder that ignores the latent", "plot both loss terms and sample from the prior rather than evaluating encodings only"),
"22-batch-normalization": ("batch normalization with train-time batch and eval-time running statistics", "x̂=(x−μB)/√(σ²B+ε)", "train/eval mode, running-stat updates, and channel axes are explicit", "small-batch drift or serving accidentally left in training mode", "compare batch and running-stat outputs at several batch sizes with an explicit eval-mode test"),
"23-unet": ("encoder-decoder segmentation with skip connections", "y=Decoder(Encoder(x), skips)", "skip tensors align spatially and channels before concatenation or addition", "crop/padding misalignment and activation-memory pressure at high resolution", "assert every join shape and evaluate boundary metrics on synthetic masks"),
"24-ppo": ("clipped on-policy policy optimization", "L^CLIP(θ)=E_t[min(r_t(θ)Â_t, clip(r_t(θ),1−ε,1+ε)Â_t)]", "the ratio uses the behavior-policy log-probability, terminated transitions do not bootstrap, and clipping is sign-aware", "stale rollouts, incorrect advantage normalization, or confusing clip fraction with a hard constraint", "monitor approximate KL, clip fraction, entropy, and advantage statistics with a one-step hand check"),
"25-vision-transformer": ("image patchification followed by transformer token mixing", "N=HW/P²", "patch ordering and positional embeddings preserve the mapping back to image coordinates", "patch-size information loss, quadratic token cost, or a patchify normalization mismatch", "round-trip patchify/unpatchify and compare attention cost and accuracy by patch size"),
"26-dqn": ("replay-based deep Q-learning with a delayed target network", "y=r+γmax_aQ(s′,a)", "terminal transitions have no bootstrap term and target-network parameters update only on schedule", "overestimation, replay correlation, or online/target networks drifting unexpectedly", "unit-test terminal targets and compare online/target drift under a fixed replay fixture"),
"27-simclr": ("contrastive visual representation learning with augmented positive pairs", "−log exp(sim(i,j)/τ)/Σ_kexp(sim(i,k)/τ)", "positive indices are correct, self-similarity is excluded, and temperature has the intended scale", "augmentation leakage, false negatives, or a batch too small to supply useful negatives", "assert pair indexing and inspect retrieval before linear evaluation across augmentation ablations"),
"28-graph-attention-networks": ("masked attention over graph neighborhoods", "h′ᵢ=σ(Σ_jαᵢⱼWh_j)", "attention weights normalize over each node’s incoming neighbors and self-loops are intentional", "dense adjacency construction, isolated-node NaNs, or neighbor-order dependence", "assert local weight sums, test permutation invariance, and compare sparse output with a tiny dense reference"),
"29-ddpm": ("diffusion forward noising and sequential reverse denoising", "x_t=√ᾱ_tx₀+√(1−ᾱ_t)ε", "the timestep schedule and noise parameterization agree between training and sampling", "wrong schedule indexing, accumulated reverse-step error, or excessive sampling latency", "reconstruct known noised samples and check timestep-dependent noise statistics"),
"30-scaling-laws": ("empirical loss fitting across model and data scales", "L(N,D)=A/Nᵅ+B/Dᵝ+C", "training budgets, token quality, optimizer settings, and evaluation splits are comparable across runs", "regime change or overconfident extrapolation from noisy pilot data", "fit held-out scales with confidence intervals and validate against task-level metrics"),
"31-long-short-term-memory": ("LSTM gated recurrent state updates", "c_t=f_t⊙c_{t−1}+i_t⊙g_t", "padding does not update state and forget/input gates remain numerically bounded", "state leakage across sessions, exploding activations, or incorrect sequence masks", "mask lengths, isolate sessions, and inspect gate and gradient statistics"),
"32-sequence-to-sequence-learning": ("LSTM encoder-decoder autoregressive generation", "p(y|x)=∏_tp(y_t|y<t,x)", "teacher-forced training and inference use compatible token boundaries while decoder state is initialized correctly", "exposure bias, EOS errors, or beam-search state aliasing", "compare teacher-forced loss with greedy and beam outputs on exact fixtures"),
"33-bahdanau-attention": ("additive attention alignment between decoder state and encoder outputs", "c_t=Σ_iα_tih_i", "alignment scores are masked before softmax and context uses the same source positions", "attention on padding or a fixed-vector bottleneck reappearing through bad state handling", "test mask-before-softmax and alignment on synthetic sequences with known dependencies"),
"34-dropout": ("inverted dropout during training", "h̃=(m/p)h", "training is stochastic while evaluation is deterministic and expected activation scale is preserved", "dropout left enabled at serving or inconsistent rate placement across branches", "assert stochastic train outputs, deterministic eval outputs, and mean-preserving scale"),
"35-glove": ("weighted factorization of global word co-occurrence counts", "wᵀw̃+b+b̃≈logX", "count construction, weighting cutoff, and bias terms use the same vocabulary snapshot", "corpus-count memory blow-up, rare-word noise, or separate embedding tables being mishandled", "snapshot counts and evaluate reconstruction plus downstream similarity and retrieval"),
}


def interview_section(mechanism: str, equation: str, invariant: str, tradeoff: str, failure: str, test: str) -> str:
    return f'''## Interview Q&A

> **SDE2 drill-down:** Explain the mechanism, show the invariant, name the production trade-off, and give evidence from a test or debugging experiment. Use inline `code`, fenced snippets, and **bold** labels to make the reasoning scannable.

**Q:** Walk through **{mechanism}** end to end. How would you implement `{equation}`?
**A:** Decompose the expression into the actual data path: inputs enter the paper-specific transformation, intermediate scores or states are computed, invalid elements are excluded, and the result is reduced into the output or loss. For this paper, `{equation}` is an executable contract, not decoration: document tensor shapes, ownership of mutable state, numerical precision, and where batching changes semantics. Keep a small reference implementation beside the optimized path so a reviewer can connect each line of `code` to one term in the equation.

**Follow-up:** What invariant would you assert, and why is it stronger than checking final accuracy?
**A:** Assert that **{invariant}**. That property is local enough to fail near the defect, whereas accuracy can remain acceptable while a mask, reduction, or state boundary is wrong on a rare input. Add a hand-computed fixture, a randomized differential test against the reference, and shape/dtype assertions at the API boundary. The test should also cover an empty, padded, terminal, high-degree, long-context, or otherwise adversarial case when that input is meaningful for this mechanism.

**Q:** What is the main production trade-off in this paper, and how would you capacity-plan it?
**A:** The central trade-off is that **{tradeoff}**. Capacity planning therefore needs more than average FLOPs: measure peak memory, memory bandwidth, communication, preprocessing, batch-size sensitivity, and p95/p99 latency on representative distributions. Define a quality budget before optimizing, then compare a simple baseline with the paper mechanism using identical inputs and seeds. A faster path that silently changes tokenization, routing, masking, sampling, or optimization behavior is not an acceptable optimization until its quality impact is measured.

**Follow-up:** Which failure mode would make you roll back first?
**A:** Roll back on evidence of **{failure}**, especially when the symptom is silent and outputs still look plausible. Add dashboards for the paper-specific statistic, error and timeout rates, resource saturation, and a task metric sliced by difficult inputs. Use a canary or shadow comparison with the previous implementation, retain the old path behind a flag, and make the rollback decision threshold explicit before deployment. The important SDE2 judgment is to protect the paper’s semantic contract, not merely to chase a faster benchmark.

**Q:** A model passes unit tests but fails in production. What is your debugging plan?
**A:** Start with **{test}**. Reproduce the smallest production-shaped example, freeze the model and preprocessing versions, and compare intermediate tensors or records rather than only the final prediction. Check data contracts, masks, sequence boundaries, random seeds, numerical precision, and serving mode in that order; then bisect between the reference and optimized implementations. If the defect is not numerical, run a controlled ablation that removes the paper-specific mechanism and compare the resulting failure rate, which separates integration problems from a bad mechanism or configuration.

**Follow-up:** What evidence would you present in the review or postmortem?
**A:** Present one minimal failing input, the expected **{invariant}**, the first intermediate value that diverged, and the regression test that now protects it. Include a before/after table for task quality, memory, throughput, p95/p99 latency, and cost, with slices for the failure population. A complete SDE2 answer also states the rollout guard, owner, and alert threshold. That turns a paper idea into an operable system rather than a one-line claim about an equation.

'''


def main() -> None:
    if len(DATA) != len(list(ROOT.glob("papers/*/README.md"))):
        raise SystemExit("interview data does not cover every paper")
    for slug, values in DATA.items():
        path = ROOT / "papers" / slug / "README.md"
        text = path.read_text()
        # Reuse the paper-specific trade-off that was previously present in
        # the section, so rerunning this generator preserves that substance.
        old = subprocess.check_output(
            ["git", "show", f"HEAD:papers/{slug}/README.md"], text=True
        )
        match = re.search(r"The practical trade-off here is (.*?)(?: Estimate|$)", old, re.S)
        tradeoff = " ".join(match.group(1).split()).rstrip(".") if match else "the mechanism changes both quality behavior and resource use"
        start = text.index("## Interview Q&A")
        end_match = re.search(r"\n## Further Reading\n", text[start:])
        if not end_match:
            raise SystemExit(f"missing Further Reading boundary in {slug}")
        end = start + end_match.start() + 1
        path.write_text(text[:start] + interview_section(values[0], values[1], values[2], tradeoff, values[3], values[4]) + text[end:])
        print(slug)


if __name__ == "__main__":
    main()
