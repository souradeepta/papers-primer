"""Minimal LoRA (Low-Rank Adaptation) linear layer, runnable smoke test.

Mirrors Hu et al. 2021 (arXiv:2106.09685), section 4.1: a frozen
pretrained weight matrix W0 plus a trainable low-rank update BA, so the
forward pass is

    h = W0 x + (alpha / r) * B A x

B is zero-initialized so the update BA is exactly zero at the start of
training (paper, section 4.1: "We use a random Gaussian initialization
for A and zero for B, so ∆W = BA is zero at the beginning of training").
A gets a random Gaussian init, matching a standard nn.Linear-style init.
"""

# Reading guide: follow the named helpers in data-flow order, then inspect the
# assertions at the bottom. Change one toy input at a time and rerun the file.
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen base linear layer with a trainable low-rank update.

    forward(x) = base(x) + (alpha / r) * (x @ A^T @ B^T)

    A has shape (r, in_features), B has shape (out_features, r), so the
    product BA reconstructs a full (out_features, in_features) update
    matrix without ever materializing it -- only A and B are stored and
    trained.
    """

    def __init__(self, in_features: int, out_features: int, r: int, alpha: int | None = None):
        super().__init__()
        self.r = r
        self.alpha = alpha if alpha is not None else r
        self.scaling = self.alpha / self.r

        # Frozen pretrained weight W0, shape (out_features, in_features).
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.base.weight.requires_grad_(False)

        # Trainable low-rank decomposition: A is (r, in), B is (out, r).
        self.A = nn.Parameter(torch.randn(r, in_features) * (1.0 / in_features ** 0.5))
        self.B = nn.Parameter(torch.zeros(out_features, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = x @ self.A.T @ self.B.T
        return base_out + self.scaling * lora_out

    def merge(self) -> torch.Tensor:
        """Effective merged weight W0 + (alpha/r) BA -- what a deployment
        would compute once and store, giving zero added inference latency
        versus a normally fine-tuned model (paper, section 4.1)."""
        return self.base.weight + self.scaling * (self.B @ self.A)


if __name__ == "__main__":
    torch.manual_seed(0)

    # Representative attention-projection-sized shapes, scaled down from a
    # real large model (e.g. GPT-3 175B uses d_model=12288) so this runs
    # instantly on CPU.
    in_features, out_features, r = 64, 32, 4
    layer = LoRALinear(in_features, out_features, r=r, alpha=8)

    # 1. At initialization, B=0 so the LoRA update contributes nothing --
    #    the forward pass must exactly match the frozen base layer alone.
    x = torch.randn(3, in_features)
    out = layer(x)
    base_only = layer.base(x)
    assert torch.allclose(out, base_only), "LoRA update should be exactly zero at init (B=0)"
    print("ok: forward pass matches frozen base exactly at init (B zero-initialized)")

    # 2. Only A and B should be trainable; the base W0 must stay frozen --
    #    this is the entire point of LoRA (train a tiny side path, freeze
    #    the pretrained weights).
    trainable = {n for n, p in layer.named_parameters() if p.requires_grad}
    assert trainable == {"A", "B"}, f"expected only A,B trainable, got {trainable}"
    assert not layer.base.weight.requires_grad, "base weight W0 must be frozen"
    print(f"ok: only {sorted(trainable)} are trainable; base W0 is frozen")

    # 3. Parameter count check: A + B should be far fewer than W0.
    base_params = layer.base.weight.numel()
    lora_params = layer.A.numel() + layer.B.numel()
    reduction = base_params / lora_params
    assert lora_params < base_params, "LoRA params should be far fewer than base params"
    print(
        f"ok: W0 has {base_params} params, A+B have {lora_params} "
        f"({reduction:.1f}x fewer trainable params)"
    )

    # 4. After "training" (perturb B so the update becomes nonzero),
    #    merging W0 + (alpha/r) BA into a single dense matrix must
    #    reproduce the exact same forward pass -- this is the
    #    zero-extra-inference-latency trick (paper, section 4.1: "we can
    #    explicitly compute and store W = W0 + BA and perform inference
    #    as usual").
    with torch.no_grad():
        layer.B.copy_(torch.randn_like(layer.B))
    unmerged_out = layer(x)
    merged_weight = layer.merge()
    merged_out = x @ merged_weight.T
    assert torch.allclose(unmerged_out, merged_out, atol=1e-5), (
        "merged W0+(alpha/r)BA forward pass should match unmerged LoRA forward pass"
    )
    print("ok: merged weight W0 + (alpha/r)*BA reproduces the unmerged forward pass exactly")
