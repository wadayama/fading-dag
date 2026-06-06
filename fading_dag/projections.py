"""Projection operators for constrained optimization on linear Gaussian DAG parameters.

These projections are intended to be applied inside `torch.no_grad()` blocks
during projected gradient ascent/descent. They return new tensors (do not
modify input in place); the caller is responsible for using `tensor.copy_()`
or similar when in-place semantics are desired.

Currently provided:
- `project_frobenius_ball(A, P)`: project a single matrix onto {X : ||X||_F^2 <= P}.
- `project_total_power(params, P)`: project a list of matrices onto
  {{A_m} : sum_m ||A_m||_F^2 <= P}, by uniform rescaling.

These are the same numerical primitives as in `gaussian_dag.projections`;
they are vendored here so this library is fully self-contained (no
`gaussian-dag` runtime dependency).
"""

from __future__ import annotations

import torch


def project_frobenius_ball(A: torch.Tensor, P: float) -> torch.Tensor:
    """Project A onto the Frobenius ball {X : ||X||_F^2 <= P}.

    Formula:
        A_proj = A * min(1, sqrt(P) / ||A||_F).

    The projection is exact (in the Euclidean sense) for the Frobenius-ball
    constraint, and preserves the direction (matrix is rescaled uniformly).

    Args:
        A: Complex or real matrix (any shape; norm is taken as Frobenius).
        P: Positive power budget.

    Returns:
        New tensor; same shape and dtype as A.
    """
    if P <= 0:
        raise ValueError(f"Power budget P must be positive, got {P}")
    norm = torch.linalg.norm(A)
    sqrt_P = torch.sqrt(torch.tensor(P, dtype=norm.dtype, device=norm.device))
    # Scale factor: min(1, sqrt(P) / ||A||_F).
    scale = torch.where(norm <= sqrt_P, torch.ones_like(norm), sqrt_P / norm)
    return A * scale


def project_total_power(
    params: list[torch.Tensor],
    P: float,
) -> list[torch.Tensor]:
    """Project a list of matrices onto sum_m ||A_m||_F^2 <= P (uniform rescaling).

    Formula:
        A_m_proj = A_m * min(1, sqrt(P) / sqrt(sum_m ||A_m||_F^2)).

    All matrices are rescaled by the same factor, preserving the
    relative magnitudes (which is the Euclidean projection of the stacked
    vector onto a single ball).

    Args:
        params: List of complex or real tensors (any shapes).
        P: Positive total power budget.

    Returns:
        List of new tensors (same length and shapes as `params`).
    """
    if P <= 0:
        raise ValueError(f"Power budget P must be positive, got {P}")
    if len(params) == 0:
        raise ValueError("params must be a non-empty list.")
    total_sq = sum((torch.linalg.norm(p) ** 2) for p in params)
    sqrt_total = torch.sqrt(total_sq)
    sqrt_P = torch.sqrt(torch.tensor(P, dtype=sqrt_total.dtype, device=sqrt_total.device))
    scale = torch.where(sqrt_total <= sqrt_P, torch.ones_like(sqrt_total), sqrt_P / sqrt_total)
    return [p * scale for p in params]
