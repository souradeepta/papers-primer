"""Toy RAG-Sequence: retrieve documents then marginalize generator answers."""
import torch
def main():
 q=torch.tensor([1.,0.]); docs=torch.tensor([[1.,0.],[.5,.5],[0.,1.]])
 scores=docs@q; top=scores.topk(2).indices; weights=scores[top].softmax(0)
 # p(answer yes/no | document); top document favors yes, second favors no.
 likelihood=torch.tensor([[.9,.1],[.25,.75]])
 answer=(weights[:,None]*likelihood).sum(0)
 changed=(torch.tensor([.05,.95])*weights[0]+likelihood[1]*weights[1])
 print('top docs:',top.tolist(),'weights:',weights.tolist(),'answer:',answer.tolist())
 assert torch.allclose(answer.sum(),torch.tensor(1.)) and answer.argmax().item()==0
 assert changed.argmax().item()==1
 print('ok: retriever-weighted marginal is normalized and evidence can change answer ranking')
if __name__=='__main__':main()
