"""Per-realization rate-function evaluator for fading multi-terminal regions.

Batched analog of ``cmi_dag.rate_region``: each rate function

    f_T = sum_n alpha_n * I(V_{A_n}; V_{B_n} | V_{C_n})

is evaluated *per channel realization*, producing a tensor of shape ``(B,)``
rather than a scalar. Aggregation (ergodic E[f_T], outage Pr[f_T < R_T], etc.)
is the caller's responsibility, using ``fading_dag.outage`` helpers.

The ``Summand`` type and the linearity structure follow ``cmi_dag.rate_region``
verbatim; the only change is the underlying conditional-MI evaluator,
``fading_dag.information.conditional_mutual_information_from_k``, which is
itself batched.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from fading_dag.information import conditional_mutual_information_from_k

# One summand alpha_{T,n} I(V_{A_n}; V_{B_n} | V_{C_n}) of the rate function f_T.
Summand = tuple[float, Sequence[int], Sequence[int], Sequence[int]]


def evaluate_rate_functions(
    K: dict[tuple[int, int], torch.Tensor],
    inequalities: Sequence[Sequence[Summand]],
    *,
    jitter: float = 0.0,
) -> list[torch.Tensor]:
    """Evaluate the rate functions ``{f_T}`` for a fading multi-terminal region.

    Per channel realization,

        f_T^{(b)} = sum_n alpha_n * I^{(b)}(V_{A_n}; V_{B_n} | V_{C_n}),

    with the per-realization conditional MI delivered by
    ``conditional_mutual_information_from_k``.

    Args:
        K: Batched canonical K-blocks produced by
            ``fading_dag.compute_k_blocks_multiroot``. Each block has
            shape ``(B, d_j, d_k)``.
        inequalities: Sequence of rate functions; each rate function is a
            sequence of ``Summand`` tuples ``(alpha, A, B, C)`` evaluating to

                f_T = sum_n alpha_n * I(V_{A_n}; V_{B_n} | V_{C_n}).

            ``alpha`` may be of either sign.
        jitter: Optional diagonal jitter forwarded to every conditional MI
            evaluation.

    Returns:
        List of length ``len(inequalities)``; each entry is a real tensor of
        shape ``(B,)`` (nats), differentiable through K.

    Raises:
        ValueError: if any rate function has zero summands.
    """
    out: list[torch.Tensor] = []
    for summands in inequalities:
        f_T: torch.Tensor | None = None
        for alpha, A, B, C in summands:
            term = alpha * conditional_mutual_information_from_k(
                K, A, B, C, jitter=jitter
            )
            f_T = term if f_T is None else f_T + term
        if f_T is None:
            raise ValueError("Each rate function must have at least one summand.")
        out.append(f_T)
    return out
