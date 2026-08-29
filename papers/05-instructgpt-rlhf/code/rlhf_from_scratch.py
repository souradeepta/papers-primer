"""Two minimal, runnable smoke tests for InstructGPT's RLHF mechanisms.

Mirrors Ouyang et al. 2022 (arXiv:2203.02155), "Training language models
to follow instructions with human feedback":

1. Reward model (paper section 3.5, eq. 1): a scalar head trained on
   pairwise human preferences with the ranking loss

       loss(theta) = -E[log(sigmoid(r_theta(x, y_w) - r_theta(x, y_l)))]

   where y_w is the labeler-preferred completion and y_l is the
   dispreferred one, for the same prompt x.

2. The KL-penalized PPO reward (paper section 3.5):

       total_reward = r_theta(x, y) - beta * log(pi_RL(y|x) / pi_SFT(y|x))

   which subtracts a per-token KL penalty against the frozen SFT/reference
   policy so the RL policy can't drift arbitrarily far from the model that
   produced the human-labeled data just to chase reward-model score.

Both are toy-scale (random frozen features, small dims) so this runs on
CPU in well under a second -- not a faithful reproduction of GPT-3 scale.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardHead(nn.Module):
    """Scalar reward head on top of frozen features -- stands in for the
    paper's reward model, which replaces the unembedding layer of a
    pretrained/SFT-fine-tuned GPT with a linear layer projecting to a
    single scalar per (prompt, completion) pair (paper, section 3.5)."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


def pairwise_ranking_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    """-log(sigmoid(r_chosen - r_rejected)), averaged over the batch.

    Paper eq. 1 (section 3.5), specialized to K=2 responses per prompt
    (the paper trains on all C(K,2) pairs from K=4-9 ranked responses per
    prompt as a single batch element; this is the pairwise term that sum
    reduces to when K=2).
    """
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def ppo_kl_penalized_reward(
    r_theta: torch.Tensor,
    logprob_policy: torch.Tensor,
    logprob_ref: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """total_reward = r_theta(x,y) - beta * log(pi_RL(y|x) / pi_SFT(y|x))

    (paper, section 3.5). logprob_policy is the *current, evolving* RL
    policy pi_RL's log-probability of the sampled completion y; logprob_ref
    is the *frozen* SFT/reference policy pi_SFT's log-probability of that
    same completion. Their difference is the (sample estimate of the)
    per-episode KL term.
    """
    kl = logprob_policy - logprob_ref
    return r_theta - beta * kl


if __name__ == "__main__":
    torch.manual_seed(0)

    # --- 1. Reward model: train a linear head on frozen random features
    # to prefer the "chosen" completion over the "rejected" one, purely
    # from pairwise comparisons -- no absolute labels, exactly as the
    # paper's labelers only ever rank completions relative to each other.
    feature_dim = 16
    n_pairs = 32
    # Two "completions" per prompt, represented as frozen random feature
    # vectors, with a synthetic ground truth: completion A is preferred
    # whenever its features sum to a larger value (a stand-in for "actual
    # human-judged quality" that the reward head has to learn to predict).
    feats_a = torch.randn(n_pairs, feature_dim)
    feats_b = torch.randn(n_pairs, feature_dim)
    a_is_chosen = feats_a.sum(dim=-1) > feats_b.sum(dim=-1)
    chosen = torch.where(a_is_chosen.unsqueeze(-1), feats_a, feats_b)
    rejected = torch.where(a_is_chosen.unsqueeze(-1), feats_b, feats_a)

    reward_model = RewardHead(feature_dim)
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.05)

    initial_loss = pairwise_ranking_loss(
        reward_model(chosen), reward_model(rejected)
    ).item()
    for _ in range(200):
        optimizer.zero_grad()
        loss = pairwise_ranking_loss(reward_model(chosen), reward_model(rejected))
        loss.backward()
        optimizer.step()
    final_loss = loss.item()

    assert final_loss < initial_loss, (
        f"reward model should learn to separate chosen/rejected: "
        f"initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f}"
    )
    with torch.no_grad():
        chosen_beats_rejected = (
            reward_model(chosen) > reward_model(rejected)
        ).float().mean().item()
    assert chosen_beats_rejected > 0.9, (
        f"trained reward model should rank chosen > rejected on most pairs, "
        f"got {chosen_beats_rejected:.2%}"
    )
    print(
        f"ok: reward model loss {initial_loss:.4f} -> {final_loss:.4f} "
        f"after training; ranks chosen > rejected on {chosen_beats_rejected:.0%} of pairs"
    )

    # --- 2. KL-penalized PPO reward: verify the penalty term behaves as
    # the paper describes -- zero when the policy hasn't moved from the
    # reference, and growing (shrinking total reward) as the policy's
    # log-probability diverges from the frozen SFT/reference policy's,
    # holding the raw reward-model score r_theta fixed.
    r_theta = torch.tensor(2.0)  # fixed reward-model score for one sampled completion
    logprob_ref = torch.tensor(-3.0)  # frozen SFT policy's log-prob of that completion
    beta = 0.2

    # No drift: policy matches reference exactly -> KL term is 0 -> total
    # reward equals the raw reward-model score.
    reward_no_drift = ppo_kl_penalized_reward(r_theta, logprob_ref, logprob_ref, beta)
    assert torch.isclose(reward_no_drift, r_theta), (
        "KL penalty should vanish when policy matches the reference exactly"
    )

    # Increasing drift: as the policy assigns higher and higher
    # log-probability to this completion relative to the frozen reference
    # (i.e. it increasingly favors an output the SFT model found
    # unlikely), the penalty should grow and total reward should fall
    # monotonically, even though the raw reward-model score is unchanged.
    drift_logprobs = [logprob_ref, logprob_ref + 1.0, logprob_ref + 2.0, logprob_ref + 4.0]
    total_rewards = [
        ppo_kl_penalized_reward(r_theta, lp, logprob_ref, beta).item() for lp in drift_logprobs
    ]
    assert all(
        total_rewards[i] > total_rewards[i + 1] for i in range(len(total_rewards) - 1)
    ), f"total reward should strictly decrease as KL drift grows, got {total_rewards}"
    print(
        f"ok: KL-penalized reward falls monotonically as policy drifts from "
        f"reference: {[round(r, 3) for r in total_rewards]} "
        f"(raw reward-model score fixed at {r_theta.item()})"
    )
