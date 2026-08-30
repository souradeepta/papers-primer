"""DPO on one preference pair: optimize a policy against a frozen reference."""
from __future__ import annotations
import torch

def main() -> None:
    torch.manual_seed(9)
    reference = torch.tensor([0.2, -0.1, 0.4])  # fixed logits: chosen=0, rejected=1
    policy = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
    opt = torch.optim.SGD([policy], lr=0.2)
    beta = 0.5
    def margin() -> torch.Tensor:
        logp, logref = policy.log_softmax(0), reference.log_softmax(0)
        return (logp[0]-logref[0]) - (logp[1]-logref[1])
    before = margin().item()
    for _ in range(80):
        opt.zero_grad(); loss = -torch.nn.functional.logsigmoid(beta * margin()); loss.backward(); opt.step()
    after = margin().item()
    print(f"relative chosen-minus-rejected margin: {before:.4f} -> {after:.4f}")
    assert after > before + 1.0
    print("ok: DPO gradient steps increase the preferred response's relative log-probability")
if __name__ == "__main__": main()
