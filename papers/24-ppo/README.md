# Proximal Policy Optimization Algorithms (PPO)

## TL;DR

PPO is a policy-gradient algorithm that reuses rollout data for several
minibatch epochs while discouraging an updated policy from moving too far from
the policy that collected that data. Its common clipped surrogate objective
limits the benefit of changing an action probability beyond a small range. PPO
is simpler than trust-region methods and became a common reinforcement-learning
baseline. It does not guarantee safe behavior, monotonic improvement, or sample
efficiency in every environment.

## Fun Map for First Years 🧭

PPO teaches an agent from rewards but stops it from changing its behavior too wildly after one lucky lesson.

`🎮 try action → 🏆 reward signal → 📏 clip giant change → 🤖 safer learning step`

An agent tries actions, receives feedback, then improves its behavior a little. PPO prevents one surprising result from causing an enormous policy change.

An agent that discovers a good jump in a game should become more likely to jump, but not immediately change from 10% to 100% confidence from one lucky trial. PPO makes that update gradual.

💻 **CS analogy:** PPO is a rate limiter around a policy update: a promising change is allowed, but a giant jump is capped before it destabilizes the running system.

## Math Playground 🧮

The essential equation or rule is:

```text
min(r_tA_t, clip(r_t,1−ε,1+ε)A_t)
```

**Essential equation:** \(\min(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)\). rₜ compares the new policy’s probability of an action with the old policy’s probability; Aₜ says whether that action turned out better or worse than expected. Clipping limits rₜ to a small range around 1. It is a safety rail: one training update cannot claim a huge reward by changing its mind too drastically.

r compares a new action probability with its old probability; A says whether the action beat expectation. Clip limits r near 1 as a safety rail.

When r stays between 1−ε and 1+ε, the unclipped improvement is used. Outside that range, clipping removes the incentive for an excessively large probability change.

## Background: What Came Before 🕰️

Policy-gradient methods could learn directly from rewards but often made updates so large that a previously useful policy collapsed. Trust-region methods improved stability but needed more complicated constrained optimization. PPO was needed as a practical approximation that keeps the update guardrail simple enough for broad adoption.

PPO made policy-gradient reinforcement learning more stable and easier to use than earlier methods with delicate update constraints.

This gave practitioners a simple trust-region-like safety mechanism and became a common baseline for control and later preference optimization.

## Why It Matters

A policy maps observations to an action distribution. Standard on-policy policy
gradients collect trajectories, estimate which actions were better than expected,
and adjust probabilities. A single large update can make the new policy unlike
the policy that produced the data, invalidating the local estimate and causing
collapse. One update per sample is also data-inefficient for costly simulation
or human-feedback rollouts.

Trust Region Policy Optimization constrained policy movement with a more complex
optimization procedure. PPO proposed practical surrogate objectives usable with
ordinary minibatch stochastic gradient ascent. The paper alternates environment
sampling with multiple optimization epochs and reported a favorable balance of
simplicity, sample complexity, and wall time on Atari and simulated locomotion.
PPO later became important in RLHF pipelines, but language-model PPO adds reward
models, KL controls, token masks, and distributed systems beyond this paper.

## Core Intuition

Suppose a coach reviews yesterday's plays. If one action looked helpful, it is
reasonable to make it somewhat more likely. It is risky to turn a modest signal
into an absolute rule after one batch: yesterday's evidence was collected under
the old strategy and may not describe the consequences of a radically changed
strategy. PPO lets the coach learn from the tape repeatedly but stops rewarding
an action-probability change once it exceeds a small trust-like neighborhood.

```mermaid
flowchart LR
 O[old policy] --> R[collect rollout]
 R --> A[advantages]
 O --> Q[old action probabilities]
 A --> S[clipped surrogate objective]
 Q --> S
 S --> N[new policy]
```

## The Mechanism

For an action sampled from old policy \(\pi_{old}\), define probability ratio
\(r_t(\theta)=\pi_\theta(a_t\mid s_t)/\pi_{old}(a_t\mid s_t)\). With advantage
estimate \(\hat A_t\), PPO's clipped objective is

\[
L^{CLIP}=E_t[\min(r_t\hat A_t,
\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)].
\]

For a positive advantage, increasing an action's probability helps until the
ratio reaches \(1+\epsilon\); beyond that, clipping removes extra objective
reward. For negative advantage, decreasing probability is similarly bounded.
The minimum selects the pessimistic of unclipped and clipped terms, discouraging
the update from exploiting the surrogate far from the rollout policy.

```mermaid
flowchart TD
 T[trajectory rewards and values] --> G[returns and advantages]
 P[old log probabilities] --> R[ratio to current policy]
 G --> L[clipped policy loss]
 R --> L
 L --> U[multiple minibatch epochs]
 U --> C[collect fresh rollout]
```

![Illustrative PPO clipping](assets/ppo_clipping.gif)

Implementations commonly add a value-function regression loss and entropy bonus
to the clipped policy objective. Advantage estimates often use generalized
advantage estimation, which trades bias and variance through gamma and lambda.
Those components are important in practical actor-critic systems but should not
be confused with the core clipping equation. The GIF is illustrative, not a
paper measurement.

PPO is on-policy: rollout actions must retain old log probabilities, values,
termination flags, and observation/action preprocessing. After too many epochs
or an overly large learning rate, the ratio can still drift; clipping is not a
hard global divergence constraint. Monitor approximate KL divergence and clip
fraction. A policy can improve the surrogate while exploiting a misspecified
reward, so reward design and environment evaluation remain central.

## Practical Engineering Notes

Use a maintained implementation such as Stable-Baselines3 PPO, CleanRL, or
RLlib as a reference before modifying a custom environment. Define reset,
termination versus truncation, action bounds, reward scale, seed handling, and
observation normalization precisely. Bootstrapping a truncated episode as if it
were terminal biases return estimates; treating a true terminal state as a
truncation leaks value beyond the episode. Unit-test these transitions.

Store rollout tensors with their behavior-policy log probabilities. Recomputing
the “old” policy after parameters change silently makes the ratio one and defeats
the objective. Normalize advantages per rollout or minibatch only according to a
documented recipe, since it changes effective update scale. Log policy loss,
value loss, entropy, approximate KL, clip fraction, explained variance, episode
return, episode length, and environment errors. No one scalar establishes that
an RL run is healthy.

Parallel environments improve throughput but affect reproducibility and reward
statistics. Capture environment versions, wrappers, simulator settings, seeds,
and action repeat. Checkpoint actor, critic, optimizer, normalizers, RNG state,
and training counters. Evaluation uses fixed seeds and typically deterministic
or documented stochastic action selection; never compare runs with different
evaluation policies without labeling the difference.

For human-feedback or high-impact policies, reward is a proxy. Optimize against
held-out adversarial scenarios, apply action constraints outside the learned
policy, and maintain rollback and human oversight. A clipped update cannot make
an unsafe reward safe. RL agents may discover loopholes that yield high reward
while violating intended behavior, so inspect trajectories and define failure
metrics before deployment.

### Debugging and evaluation discipline

Start with a tiny environment where a random policy and an optimal policy are
easy to distinguish. Verify action distributions respect bounds, returns reset
at episode boundaries, and a known reward produces an expected advantage sign.
Then overfit a deterministic small problem. If PPO cannot learn that case,
changing network size or adding more workers only hides a dataflow error. Inspect
one trajectory by hand: observation, action, reward, value, return, advantage,
old log probability, ratio, and clipped term should all be explainable.

Hyperparameters interact. Rollout length determines how fresh data is and how
far credit assignment can reach; number of epochs determines reuse; minibatch
size changes optimizer noise; epsilon changes the surrogate's local tolerance.
Learning rate, entropy coefficient, value coefficient, reward scaling, gamma,
and lambda must be logged together. Scaling rewards changes advantage magnitude
and therefore policy updates even if the objective formula is unchanged. Use a
small sweep with fixed evaluation protocol rather than adopting settings from a
different environment blindly.

Separate training exploration from product behavior. Entropy encourages policy
randomness during learning, but an evaluator or deployed system may need a
deterministic argmax or a controlled sampling temperature. Report which action
selection policy produced each score. Test rare resets, invalid actions, delayed
rewards, simulator timeouts, and observation corruption. These operational cases
often dominate real-agent reliability more than average benchmark return.

If the environment has costly, irreversible, or human-facing actions, train in
an isolated simulator and enforce external permission checks at execution time.
Rate limits, action allowlists, budget caps, and manual escalation are system
controls, not rewards. Retain enough trajectory provenance to investigate a bad
outcome without collecting unnecessary sensitive data. A policy checkpoint
should never be deployed merely because its reward chart rose faster.

PPO is valuable because it makes the relationship between rollout data and
updates explicit. That clarity supports rigorous evaluation: compare equal
interaction budgets, confidence intervals over seeds, worst-case episodes, and
wall-clock cost. A single fortunate seed or an average-only plot is too weak to
claim a policy is ready for an environment with meaningful consequences.
Record failures as carefully as successes: they reveal reward loopholes,
environment assumptions, and distribution shifts that summary metrics conceal.
They are essential evidence for responsible model operation.

## Runnable Code Example

### Run it

The implementation is intentionally small and self-checking. From the repository root, use Python 3; the module docstring states the learning goal, comments identify the paper-specific calculation, and assertions verify the toy invariant.

```bash
python3 papers/24-ppo/code/clipped_objective.py
```

### Read it in order

Start with the module docstring, then follow the named helper calculations and the final assertions. The example is a dependency-light teaching implementation, not a production training system; change one input at a time and rerun it to see which invariant changes.


[`code/clipped_objective.py`](code/clipped_objective.py) calculates a positive-
advantage clipped surrogate and asserts that a ratio beyond the upper bound gains
no extra objective value.

```bash
python3 papers/24-ppo/code/clipped_objective.py
```

It demonstrates one scalar term, not an environment, neural policy, or value
function trainer.

## Common Misconceptions & Pitfalls

**“PPO forbids policy changes beyond epsilon.”** It clips objective incentive;
the actual policy can still move because parameters affect many actions.

**“PPO is off-policy because it has multiple epochs.”** Its samples come from a
recent behavior policy and must be refreshed regularly.

**“A higher reward proves correct behavior.”** Reward can be misspecified or
gamed, especially outside the training distribution.

## Interview Q&A

**Q:** What is the PPO ratio?
**A:** The current policy's action probability divided by the rollout policy's
probability for the same state-action pair.

**Q:** Why clip the objective?
**A:** To reduce incentive for overly large probability changes on fixed rollout
data while retaining simple first-order optimization.

**Q:** What is an advantage estimate?
**A:** An estimate of how much better an action was than the policy's expected
value at that state.

**Q:** Why retain old log probabilities?
**A:** They are needed to calculate the behavior-to-current policy ratio.

**Q:** Is PPO a safety mechanism?
**A:** No. It stabilizes optimization; safety needs constraints, evaluation, and
operational controls.

## Further Reading

- [Original paper](https://arxiv.org/abs/1707.06347)
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [Stable-Baselines3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
