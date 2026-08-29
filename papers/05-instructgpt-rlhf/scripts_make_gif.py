"""Generate assets/kl_drift.gif: an animation of InstructGPT/RLHF's single
most motion-worthy mechanism -- the PPO policy's output distribution
drifting away from the frozen SFT/reference policy over training steps,
and the KL divergence between them growing as it does.

This is the quantity the paper's KL penalty term is built to control:
"total_reward = r_theta(x,y) - beta * log(pi_RL(y|x) / pi_SFT(y|x))"
(paper, section 3.5). Without that penalty, nothing stops the RL policy
from concentrating all its probability mass on whatever the reward model
scores highest, even if that means abandoning the token distribution the
SFT model (and the labelers' demonstrations behind it) considered
reasonable -- a failure mode usually called "reward hacking" or
"reward-model over-optimization."

This figure is illustrative, not extracted from a trained model: it
hand-constructs a sequence of toy discrete distributions over 6 stand-in
"response categories" that interpolate from the frozen reference
distribution toward a distribution that heavily favors one
reward-preferred category, and plots both the shifting distribution and
the resulting KL(pi_RL || pi_SFT) trend at each step.

One-off generator; not part of the validated code/ smoke test.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

rng = np.random.default_rng(0)

categories = ["A", "B", "C", "D", "E", "F"]
n_cat = len(categories)

# Frozen reference (SFT) policy: a mildly peaked but fairly spread-out
# distribution over response categories -- stands in for pi_SFT(.|x).
ref_logits = np.array([1.2, 0.8, 0.5, 0.3, -0.2, -0.6])
ref_dist = np.exp(ref_logits) / np.exp(ref_logits).sum()

# Reward-favored target: the reward model scores category "A" far above
# the rest -- stands in for what unconstrained reward maximization alone
# would push the policy toward.
reward_logits = np.array([6.0, -1.0, -1.0, -1.0, -1.0, -1.0])

n_steps = 14
alphas = np.linspace(0.0, 1.0, n_steps)  # 0 = pure reference, 1 = pure reward-chasing


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / q)))


policy_dists = []
kls = []
for alpha in alphas:
    logits = (1 - alpha) * ref_logits + alpha * reward_logits
    dist = np.exp(logits) / np.exp(logits).sum()
    policy_dists.append(dist)
    kls.append(kl_divergence(dist, ref_dist))

out_path = Path(__file__).parent / "assets" / "kl_drift.gif"
out_path.parent.mkdir(exist_ok=True)

fig, (ax_bar, ax_kl) = plt.subplots(1, 2, figsize=(10, 4.5))
writer = PillowWriter(fps=1.5)

x = np.arange(n_cat)
width = 0.35

with writer.saving(fig, str(out_path), dpi=100):
    for step in range(n_steps):
        ax_bar.clear()
        ax_kl.clear()

        ax_bar.bar(x - width / 2, ref_dist, width, color="#4C72B0", label="pi_SFT (frozen reference)")
        ax_bar.bar(x + width / 2, policy_dists[step], width, color="#C44E52", label="pi_RL (PPO policy)")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(categories)
        ax_bar.set_ylim(0, 1.0)
        ax_bar.set_ylabel("probability")
        ax_bar.set_title("Policy vs. frozen reference\n(response categories, toy)", fontsize=9.5)
        ax_bar.legend(loc="upper right", fontsize=7.5)

        ax_kl.plot(range(step + 1), kls[: step + 1], color="#55A868", marker="o", markersize=4)
        ax_kl.set_xlim(0, n_steps - 1)
        ax_kl.set_ylim(0, max(kls) * 1.15)
        ax_kl.set_xlabel("PPO training step")
        ax_kl.set_ylabel("KL(pi_RL || pi_SFT)")
        ax_kl.set_title(
            "KL divergence grows as the policy\nchases reward, unconstrained",
            fontsize=9.5,
        )
        ax_kl.grid(alpha=0.3)
        ax_kl.annotate(
            f"KL={kls[step]:.2f}",
            xy=(step, kls[step]),
            xytext=(0, 8),
            textcoords="offset points",
            fontsize=8,
            ha="center",
        )

        fig.suptitle(
            "total_reward = r_theta(x,y) - beta * log(pi_RL(y|x)/pi_SFT(y|x))\n"
            "the KL term is exactly what penalizes this drift during real PPO training",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        writer.grab_frame()
    # Hold the final frame so the fully-drifted state is visible.
    for _ in range(3):
        writer.grab_frame()

plt.close(fig)
print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")
