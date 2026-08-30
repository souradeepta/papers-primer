"""RAG-Sequence retrieval and document-marginalized generation.

RAG uses a dense retriever to choose evidence from an external index, then
marginalizes a generator's answer probability across the top documents. This
small implementation retains the key paper distinction: retrieval probability
is part of the likelihood, not merely a prompt-formatting convenience.
"""

from __future__ import annotations

import torch


def retrieve(query: torch.Tensor, document_embeddings: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k document ids and normalized dense-retriever probabilities."""
    similarity = document_embeddings @ query
    top_scores, document_ids = similarity.topk(top_k)
    return document_ids, top_scores.softmax(dim=0)


def rag_sequence_probability(
    retriever_probability: torch.Tensor, generator_probability: torch.Tensor
) -> torch.Tensor:
    """Implement p(y|x) = sum_z p_eta(z|x) p_theta(y|x,z) for one generated token."""
    return (retriever_probability[:, None] * generator_probability).sum(dim=0)


def main() -> None:
    # Rows form a tiny frozen dense index; rows of likelihoods are a generator
    # distribution over answer tokens yes/no conditioned on each retrieved doc.
    query = torch.tensor([1.0, 0.0])
    index = torch.tensor([[1.0, 0.0], [0.7, 0.3], [0.0, 1.0], [-1.0, 0.0]])
    document_ids, retrieval_probability = retrieve(query, index, top_k=2)
    generator_probability = torch.tensor([[0.90, 0.10], [0.25, 0.75]])
    answer_probability = rag_sequence_probability(retrieval_probability, generator_probability)

    print(f"retrieved ids: {document_ids.tolist()}; weights: {[round(x, 3) for x in retrieval_probability.tolist()]}")
    print(f"marginal answer distribution yes/no: {[round(x, 3) for x in answer_probability.tolist()]}")
    assert document_ids.tolist() == [0, 1]
    assert torch.allclose(answer_probability.sum(), torch.tensor(1.0))
    assert answer_probability.argmax().item() == 0
    print("ok: RAG combines generator probabilities using retriever confidence")


if __name__ == "__main__":
    main()
