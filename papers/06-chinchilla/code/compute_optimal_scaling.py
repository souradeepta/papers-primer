"""A tiny Chinchilla-style fixed-compute allocation experiment.

Hoffmann et al. model pre-training loss as an irreducible term plus a
model-size term and a finite-data term.  Under the common approximation
that training compute C is proportional to 6 * N * D, spending a fixed C
means a larger parameter count N leaves fewer training tokens D.  This
script uses that shape to show that both extremes lose: the lowest loss is
near a balanced parameter/data allocation.

It is a teaching simulation, not a reproduction of Chinchilla's fitted
coefficients or a recommendation for selecting a production model size.
"""
import math


def toy_loss(parameters: float, tokens: float) -> float:
    """A scaled version of L(N,D) = E + A/N^alpha + B/D^beta.

    Constants are chosen so that a small CPU-only sweep is readable.  Equal
    exponents make the optimum occur where parameter and token allocations
    are balanced after their units are normalized.
    """
    irreducible = 1.0
    model_limited = 100.0 / math.sqrt(parameters)
    data_limited = 100.0 / math.sqrt(tokens)
    return irreducible + model_limited + data_limited


def main() -> None:
    # Fixed normalized compute: C ~= N * D.  The factor 6 used for dense
    # Transformer training is a constant here and does not change the argmin.
    compute = 1_000_000.0
    candidates = [10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000]
    rows = []
    for parameters in candidates:
        tokens = compute / parameters
        rows.append((toy_loss(parameters, tokens), parameters, tokens))

    best_loss, best_parameters, best_tokens = min(rows)
    print("fixed normalized compute = N * D = 1,000,000")
    print("parameters  tokens     toy loss")
    for loss, parameters, tokens in rows:
        marker = "  <-- lowest" if parameters == best_parameters else ""
        print(f"{parameters:>10,.0f}  {tokens:>8,.0f}  {loss:.4f}{marker}")

    # With this symmetric toy loss and N*D fixed, AM-GM predicts N=D=1000.
    assert best_parameters == 1_000 and best_tokens == 1_000
    assert best_loss < toy_loss(10, compute / 10)
    assert best_loss < toy_loss(10_000, compute / 10_000)
    print("ok: at fixed compute, neither an oversized model nor too little data wins")


if __name__ == "__main__":
    main()
