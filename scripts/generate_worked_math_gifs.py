"""Generate one compact, paper-specific equation walkthrough GIF per explainer."""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/papers-primer-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter


ROOT = Path(__file__).parent.parent

CARDS = {
    "01-attention-is-all-you-need": ("Attention", "softmax(QKᵀ / √d)", "compare queries and keys", "weighted values"),
    "02-bert": ("BERT MLM", "−log p(xᵢ | x\u0304)", "hide selected tokens", "recover context bidirectionally"),
    "03-gpt3-few-shot-learners": ("GPT-3", "p(xₜ | x<ₜ)", "show task examples", "continue the pattern"),
    "04-lora": ("LoRA", "W′ = W + BA", "freeze W", "learn a low-rank update"),
    "05-instructgpt-rlhf": ("RLHF", "reward − β KL(π || πref)", "score a response", "improve while limiting drift"),
    "06-chinchilla": ("Chinchilla", "C ≈ 6ND", "fix compute", "balance model and data"),
    "07-flashattention": ("FlashAttention", "softmax(QKᵀ)V", "stream tiles", "correct running statistics"),
    "08-roformer-rope": ("RoPE", "R(m)ᵀR(n) = R(n−m)", "rotate by position", "preserve relative phase"),
    "09-dpo": ("DPO", "logit σ(β log odds)", "compare preferred pair", "directly update the policy"),
    "10-switch-transformer": ("Switch", "y = Expert[argmax p(x)](x)", "route each token", "specialize capacity"),
    "11-sentencepiece": ("SentencePiece", "argmax_seg ∏ p(piece)", "score candidate pieces", "choose the best segmentation"),
    "12-t5": ("T5", "text → text", "corrupt spans", "generate sentinel targets"),
    "13-rag": ("RAG", "p(y|x)=Σ_z p(z|x)p(y|x,z)", "retrieve passages", "marginalize evidence"),
    "14-chain-of-thought": ("CoT", "argmax_y Σ_k 1[y=y_k]", "sample traces", "vote across answers"),
    "15-pagedattention-vllm": ("PagedAttention", "logical → physical blocks", "allocate KV pages", "share prefixes safely"),
    "16-word2vec": ("word2vec", "log σ(vᵀv′)+Σ log σ(−vᵀvₙ)", "sample a context", "separate negatives"),
    "17-adam": ("Adam", "θ ← θ − α m̂/(√v̂+ε)", "track mean and variance", "normalize the step"),
    "18-resnet": ("ResNet", "y = F(x)+x", "learn a residual", "carry the identity path"),
    "19-gan": ("GAN", "min_G max_D V(D,G)", "discriminate real/fake", "train the generator"),
    "20-clip": ("CLIP", "sim(I,T)=Ĩ·T̃", "embed paired data", "raise diagonal similarity"),
    "21-vae": ("VAE", "ELBO = E_q[log p(x|z)] − KL(q||p)", "sample latent noise", "reconstruct and regularize"),
    "22-batch-normalization": ("BatchNorm", "x̂=(x−μ_B)/√(σ²_B+ε)", "measure the batch", "normalize then affine-transform"),
    "23-unet": ("U-Net", "y = Decoder(Encoder(x), skips)", "compress context", "restore precise boundaries"),
    "24-ppo": ("PPO", "min(rA, clip(r,1−ε,1+ε)A)", "measure policy ratio", "clip oversized updates"),
    "25-vision-transformer": ("ViT", "N = HW/P² patches", "flatten image patches", "attend globally"),
    "26-dqn": ("DQN", "y=r+γ max_a Q(s′,a)", "sample replay", "fit the Bellman target"),
    "27-simclr": ("SimCLR", "−log exp(sim(i,j)/τ)/Σ_k exp(sim(i,k)/τ)", "augment one image twice", "pull positives together"),
    "28-graph-attention-networks": ("GAT", "h′ᵢ=σ(Σ_j αᵢⱼWh_j)", "score neighbors", "aggregate weighted messages"),
    "29-ddpm": ("DDPM", "x_t=√ᾱ_t x₀+√(1−ᾱ_t)ε", "add known noise", "learn to reverse it"),
    "30-scaling-laws": ("Scaling laws", "L(N,D)=A/Nᵅ+B/Dᵝ+C", "fit small runs", "plan the next budget"),
    "31-long-short-term-memory": ("LSTM", "c_t=f_t⊙c_{t−1}+i_t⊙g_t", "retain or write memory", "expose a gated state"),
    "32-sequence-to-sequence-learning": ("Seq2Seq", "p(y|x)=∏_t p(y_t|y<t,x)", "encode the source", "decode until EOS"),
    "33-bahdanau-attention": ("Bahdanau attention", "c_t=Σ_i α_ti h_i", "score source positions", "blend relevant states"),
    "34-dropout": ("Dropout", "h̃=(m/p)h", "sample a mask", "train a robust subnetwork"),
    "35-glove": ("GloVe", "wᵀw̃+b+b̃≈log X", "count word contexts", "fit vector geometry"),
}


def make_gif(slug, card):
    title, equation, step_a, step_b = card
    out = ROOT / "papers" / slug / "assets" / "worked_math.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    writer = PillowWriter(fps=1.2)
    with writer.saving(fig, out, dpi=110):
        for stage, caption in (("1", step_a), ("2", step_b), ("✓", "the equation connects both steps")):
            ax.clear()
            ax.axis("off")
            ax.set_title(f"{title}: worked view", fontsize=15, weight="bold")
            ax.text(0.5, 0.62, equation, ha="center", va="center", fontsize=22,
                    bbox=dict(boxstyle="round,pad=.55", fc="#e6f1f8", ec="#37779a"))
            ax.text(0.5, 0.22, f"stage {stage} — {caption}", ha="center", va="center", fontsize=13)
            writer.grab_frame()
    plt.close(fig)


if __name__ == "__main__":
    for slug, card in CARDS.items():
        make_gif(slug, card)
