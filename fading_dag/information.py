"""Batched conditional mutual information from K-blocks (per-realization).

Batched analog of `cmi_dag.information`: with the K-blocks produced by
``fading_dag.krecursion.compute_k_blocks_multiroot`` carrying a leading
batch axis ``B`` of independent channel realizations, the conditional
mutual information evaluator returns a real tensor of shape ``(B,)`` whose
entries are per-realization log-det differences

    I^{(b)}(V_A; V_B | V_C) = log det Sigma_{A|C}^{(b)} - log det Sigma_{A|BC}^{(b)}.

Aggregation is the caller's responsibility:
- ``ergodic_capacity(I_samples) == I_samples.mean()`` for ergodic E[I],
- ``outage_probability(I_samples, R)`` for the empirical outage CDF, etc.

The Cholesky-based ``logdet_hpd`` accepts leading batch dimensions and
returns shape ``A.shape[:-2]``; with the (B, d, d) blocks of this library
that means a scalar tensor of shape ``(B,)``.

``logdet_hpd`` is the same numerical primitive as in
``cmi_dag.information``; it is vendored here (with extended batch support)
so this library is fully self-contained.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from fading_dag.krecursion import get_K, hermitianize


def logdet_hpd(A: torch.Tensor, jitter: float = 0.0) -> torch.Tensor:
    """Batched Cholesky-based log-determinant for Hermitian PD matrices.

    For Hermitian positive-definite ``A = L L^H`` with lower-triangular ``L``
    and real positive diagonal,

        log det A = 2 * sum_i log L_ii.

    The input is symmetrized by ``(A + A^H)/2`` to enforce Hermitian
    structure against floating-point drift before the Cholesky step. The
    Cholesky factorisation is performed via ``torch.linalg.cholesky_ex``;
    on failure (matrix not strictly positive-definite at any batch entry)
    this function raises a ``ValueError`` with diagnostic information.

    Supports leading batch dimensions:
    - Input shape ``A.shape == (..., d, d)``.
    - Output shape ``A.shape[:-2]``.

    Args:
        A: Hermitian positive-definite matrix or batched stack thereof.
        jitter: If > 0, replace A by ``A + jitter * I`` before factorization.

    Returns:
        Real tensor of shape ``A.shape[:-2]`` (natural log; nats convention).

    Raises:
        ValueError: if any batch entry of A (after optional jitter) is not
            strictly positive definite.
    """
    A = hermitianize(A)
    if jitter > 0.0:
        d = A.shape[-1]
        eye = torch.eye(d, dtype=A.dtype, device=A.device)
        A = A + jitter * eye
    L, info = torch.linalg.cholesky_ex(A, check_errors=False)
    # `info` has the same leading batch shape as A.shape[:-2]; nonzero entries
    # flag PD failure at that batch index.
    if torch.any(info != 0):
        bad = (info != 0).nonzero(as_tuple=False)
        first = tuple(bad[0].tolist()) if bad.numel() > 0 else ()
        raise ValueError(
            "logdet_hpd: input matrix is not Hermitian positive definite at "
            f"batch index {first} (info={int(info[first].item())}). "
            "Common remedies: (1) ensure the terminal noise covariance is "
            "strictly positive definite (the regularity assumption); "
            "(2) pass jitter>0 to logdet_hpd / "
            "conditional_mutual_information_from_k to absorb near-singularity; "
            "(3) reduce SGD step size so iterates remain in the PD cone."
        )
    diag = torch.diagonal(L, dim1=-2, dim2=-1).real
    return 2.0 * torch.log(diag).sum(dim=-1)


def _assemble(
    K: dict[tuple[int, int], torch.Tensor],
    rows: Sequence[int],
    cols: Sequence[int],
) -> torch.Tensor:
    """Stack K-blocks into a single covariance Sigma_{rows, cols}.

    Block ``(r, c)`` is ``K_{rc} = E[V_r V_c^H]``, read via the
    Hermitian-flip helper ``get_K``. All blocks are assumed to share a
    common leading batch axis (uniform 3-D layout from
    ``compute_k_blocks_multiroot``).

    Returns a tensor of shape
    ``(B, sum_{r in rows} d_r, sum_{c in cols} d_c)``.
    """
    row_strips = []
    for r in rows:
        blocks = [get_K(K, r, c) for c in cols]
        row_strips.append(torch.cat(blocks, dim=-1))
    return torch.cat(row_strips, dim=-2)


def _conditional_cov(
    K: dict[tuple[int, int], torch.Tensor],
    A: Sequence[int],
    Z: Sequence[int],
) -> torch.Tensor:
    """Conditional covariance Sigma_{A|Z} via the (batched) Schur complement.

    Returns ``Sigma_{A,A} - Sigma_{A,Z} @ Sigma_{Z,Z}^{-1} @ Sigma_{Z,A}``
    for non-empty Z, and the marginal ``Sigma_{A,A}`` when Z is empty.
    ``torch.linalg.solve`` is batched, so no explicit inverse is formed.
    """
    Sigma_AA = _assemble(K, A, A)
    if len(Z) == 0:
        return Sigma_AA
    Sigma_AZ = _assemble(K, A, Z)
    Sigma_ZZ = _assemble(K, Z, Z)
    Sigma_ZA = _assemble(K, Z, A)
    return Sigma_AA - Sigma_AZ @ torch.linalg.solve(Sigma_ZZ, Sigma_ZA)


def conditional_mutual_information_from_k(
    K: dict[tuple[int, int], torch.Tensor],
    A: Sequence[int],
    B: Sequence[int],
    C: Sequence[int] = (),
    *,
    jitter: float = 0.0,
) -> torch.Tensor:
    """Per-realization conditional mutual information I(V_A; V_B | V_C).

    Implements the log-det closed form

        I^{(b)}(V_A; V_B | V_C)
            = log det Sigma_{A|C}^{(b)} - log det Sigma_{A|BC}^{(b)},

    independently for every batch index ``b``. With an empty conditioning
    set ``C`` this reduces to the unconditional MI ``I(V_A; V_B)``.

    Args:
        K: Batched canonical K-blocks produced by
            ``fading_dag.compute_k_blocks_multiroot``. Every block has
            shape ``(B, d_j, d_k)``.
        A: Node indices of the first information set (non-empty).
        B: Node indices of the second information set (non-empty).
        C: Conditioning node indices (default empty -> unconditional MI).
        jitter: Optional diagonal jitter passed to ``logdet_hpd`` for both
            ``Sigma_{A|C}`` and ``Sigma_{A|BC}``; useful for low-SNR or
            rank-deficient regimes near the boundary of the PD cone.

    Returns:
        Real tensor of shape ``(B,)`` in nats, differentiable through K.

    Raises:
        ValueError: if A or B is empty, or if A, B, C are not pairwise
            disjoint, or if any conditional covariance fails Cholesky.
    """
    A = sorted(A)
    B = sorted(B)
    C = sorted(C)
    if len(A) == 0 or len(B) == 0:
        raise ValueError("A and B must both be non-empty.")
    all_nodes = A + B + C
    if len(all_nodes) != len(set(all_nodes)):
        raise ValueError(
            f"A, B, C must be pairwise disjoint; got A={A}, B={B}, C={C}."
        )

    Sigma_A_given_C = _conditional_cov(K, A, C)
    Sigma_A_given_BC = _conditional_cov(K, A, sorted(B + C))
    return logdet_hpd(Sigma_A_given_C, jitter=jitter) - logdet_hpd(
        Sigma_A_given_BC, jitter=jitter
    )
