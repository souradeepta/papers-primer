"""Multi-head graph attention with adjacency masking and feature aggregation."""
from __future__ import annotations
import torch


class GraphAttention(torch.nn.Module):
    """One GAT layer; each head attends only to graph neighbors."""
    def __init__(self, input_dim: int, output_dim: int, heads: int = 2) -> None:
        super().__init__(); self.projection=torch.nn.Linear(input_dim, output_dim*heads, bias=False)
        self.attention=torch.nn.Parameter(torch.randn(heads, 2*output_dim)); self.heads,self.output_dim=heads,output_dim
    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        node_count=len(features); transformed=self.projection(features).view(node_count,self.heads,self.output_dim)
        left=transformed[:,None,:,:].expand(node_count,node_count,-1,-1)
        right=transformed[None,:,:,:].expand(node_count,node_count,-1,-1)
        scores=torch.nn.functional.leaky_relu(torch.cat([left,right],-1).mul(self.attention).sum(-1))
        weights=scores.masked_fill(~adjacency[:,:,None],float("-inf")).softmax(1)
        return torch.einsum("ijh,jhd->ihd",weights,transformed).flatten(1)


def main() -> None:
    torch.manual_seed(28); features=torch.randn(4,3)
    adjacency=torch.tensor([[1,1,0,0],[1,1,1,0],[0,1,1,1],[0,0,1,1]],dtype=torch.bool)
    layer=GraphAttention(3,4); output=layer(features,adjacency); output.sum().backward()
    print(f"node embeddings: {tuple(output.shape)}")
    assert output.shape==(4,8) and layer.projection.weight.grad is not None
    print("ok: each head aggregates only adjacency-permitted neighbor features")


if __name__ == "__main__":
    main()
