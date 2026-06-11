"""Batched multi-root K-recursion for fading linear Gaussian DAGs.

Model (0-based indexing, fading extension of cmi_dag.krecursion):
    Roots r in {0, ..., K-1}:  V_r ~ CN(0, Sigma_r), mutually independent.
    Non-roots j in {K, ..., M-1}:
        V_j = sum_{i in Pa(j)} A_{ji}^{(b)} V_i + Z_j,
            Z_j ~ CN(0, Sigma_j),
    where the **per-realization** edge matrix is the product of the random
    channel sample and the deterministic controllable factor:
        A_{ji}^{(b)} = H_{ji}^{(b)} @ F_{ji},   b = 0, ..., B-1.
    The K-recursion is run once for the entire mini-batch of B independent
    channel realizations; the controllable factor F_{ji} is shared across
    realizations.

Edge representation:
    edge_mats[(j, i)] = (H_sampler, F)
    - H_sampler: Callable[[int], Tensor] returning shape (B, d_j, d_mid) of
      complex dtype. Called *exactly once* per `compute_k_blocks_multiroot`
      invocation; the resulting batch of channel matrices is reused for both
      the cross- and self-block updates.
    - F: deterministic tensor of shape (d_mid, d_i). Shared across the batch.
      Typically `requires_grad=True` so that gradients flow to the design.

Layout convention:
    - K-blocks indexed by canonical (j, k) with j >= k.
    - Blocks involving at least one non-root index are batched with leading
      batch axis B: shape (B, d_j, d_k).
    - Blocks between two root indices are *unbatched* (shape (d_r, d_r2));
      they broadcast naturally under PyTorch matmul/sum. This avoids
      duplicating root covariances B times in memory.

`hermitianize` and `get_K` are the same numerical primitives as in
`cmi_dag.krecursion` and `gaussian_dag.krecursion`; they are vendored here
so this library is fully self-contained.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

# Type alias for an edge specification: (H_sampler, F).
EdgeSpec = tuple[Callable[[int], torch.Tensor], torch.Tensor]


def hermitianize(A: torch.Tensor) -> torch.Tensor:
    """Symmetrize the trailing two dimensions of A by (A + A^H) / 2.

    Batch-safe via ``.mH`` (PyTorch's conjugate transpose on the last two dims).
    """
    return 0.5 * (A + A.mH)


def get_K(
    K: dict[tuple[int, int], torch.Tensor],
    a: int,
    b: int,
) -> torch.Tensor:
    """Return K_{ab}, applying the Hermitian flip when ``a < b``.

    K stores only canonical keys (j, k) with j >= k. For a < b, we return
    ``K[(b, a)].mH``, which is batch-safe.
    """
    if a >= b:
        return K[(a, b)]
    return K[(b, a)].mH


def _realize_edges(
    edge_mats: dict[tuple[int, int], EdgeSpec],
    batch_size: int,
) -> dict[tuple[int, int], torch.Tensor]:
    """Pre-compute A_eff[(j,i)] = H_sampler(batch_size) @ F for every edge.

    Each sampler is called *exactly once* with the requested batch size, and
    the resulting batched A_eff is reused throughout the K-recursion forward
    pass. This is required for correctness: cross- and self-block updates
    must use identical channel realizations.

    Args:
        edge_mats: per-edge 2-tuples ``(H_sampler, F)``.
        batch_size: positive integer batch size.

    Returns:
        Dict ``A_eff[(j, i)]`` with shape ``(batch_size, d_j, d_i)`` per edge.

    Raises:
        ValueError: if any edge spec is not a 2-tuple of (callable, Tensor),
            if F is not complex or its dtype differs from the sampler output,
            if the sampler output shape is inconsistent with F, or if
            batch_size <= 0.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    A_eff: dict[tuple[int, int], torch.Tensor] = {}
    for (j, i), spec in edge_mats.items():
        if not (isinstance(spec, tuple) and len(spec) == 2):
            raise ValueError(
                f"edge_mats[({j}, {i})] must be a 2-tuple (H_sampler, F); got "
                f"{type(spec).__name__}"
            )
        H_sampler, F = spec
        if not callable(H_sampler):
            raise ValueError(
                f"edge_mats[({j}, {i})][0] must be callable (a sampler); got "
                f"{type(H_sampler).__name__}"
            )
        if not isinstance(F, torch.Tensor) or F.dim() != 2:
            raise ValueError(
                f"edge_mats[({j}, {i})][1] must be a 2-D Tensor; got "
                f"{type(F).__name__}, shape "
                f"{tuple(F.shape) if isinstance(F, torch.Tensor) else 'n/a'}"
            )
        if not F.is_complex():
            raise ValueError(
                f"edge_mats[({j}, {i})][1] must be a complex tensor (the "
                f"log-det MI convention assumes circularly-symmetric complex "
                f"Gaussians); got dtype {F.dtype}"
            )
        H = H_sampler(batch_size)
        if not isinstance(H, torch.Tensor) or H.dim() != 3:
            raise ValueError(
                f"H_sampler at edge ({j}, {i}) must return a 3-D Tensor "
                f"(B, d_j, d_mid); got shape "
                f"{tuple(H.shape) if isinstance(H, torch.Tensor) else 'n/a'}"
            )
        if H.dtype != F.dtype:
            raise ValueError(
                f"H_sampler at edge ({j}, {i}) returns dtype {H.dtype}, but F "
                f"has dtype {F.dtype}; the sampler and the controllable "
                f"factor must share one complex dtype"
            )
        if H.shape[0] != batch_size:
            raise ValueError(
                f"H_sampler at edge ({j}, {i}) returned batch size "
                f"{H.shape[0]} != requested {batch_size}"
            )
        if H.shape[-1] != F.shape[0]:
            raise ValueError(
                f"H_sampler at edge ({j}, {i}) returns inner dimension "
                f"{H.shape[-1]} != F.shape[0]={F.shape[0]}"
            )
        # Broadcast: (B, d_j, d_mid) @ (d_mid, d_i) -> (B, d_j, d_i).
        A_eff[(j, i)] = H @ F
    return A_eff


def compute_k_blocks_multiroot(
    num_nodes: int,
    roots: Sequence[int],
    parents: dict[int, list[int]],
    edge_mats: dict[tuple[int, int], EdgeSpec],
    root_covs: dict[int, torch.Tensor],
    noise_covs: dict[int, torch.Tensor],
    *,
    batch_size: int,
    symmetrize_self_blocks: bool = True,
) -> dict[tuple[int, int], torch.Tensor]:
    """Compute all canonical K-blocks of a fading multi-root linear Gaussian DAG.

    Batched analog of `cmi_dag.compute_k_blocks_multiroot`: each edge
    transformation is realized B times via its sampler at function entry,
    and the K-recursion is run on the resulting batched effective matrices.
    Blocks involving at least one non-root index acquire a leading batch
    axis of size ``batch_size``; root-root blocks remain unbatched
    (broadcasting handles the asymmetry).

    Recursion:
        K_{rr}     = Sigma_r                       (r in roots)         [unbatched]
        K_{r,r'}   = 0                              (distinct roots)     [unbatched]
        K_{jk}     = sum_{i in Pa(j)} A_{ji}^{(b)} K_{ik}                [batched]
        K_{jj}     = sum_{i,i' in Pa(j)}
                      A_{ji}^{(b)} K_{ii'} (A_{ji'}^{(b)})^H + Sigma_j   [batched]

    Args:
        num_nodes: Total number of nodes M.
        roots: Indices of the root nodes. Must be exactly the prefix
            ``{0, ..., K-1}`` in topological order, with K = len(roots) and
            K < num_nodes.
        parents: ``parents[j]`` = list of parent indices for non-root j.
            Must satisfy ``i < j`` for every parent i.
        edge_mats: ``edge_mats[(j, i)] = (H_sampler, F)``. ``H_sampler`` is
            called once with ``batch_size`` per invocation of this function.
            ``F`` is a deterministic ``(d_mid, d_i)`` tensor.
        root_covs: ``root_covs[r] = Sigma_r`` (shape ``(d_r, d_r)``).
        noise_covs: ``noise_covs[j] = Sigma_j`` (shape ``(d_j, d_j)``).
        batch_size: Number of channel realizations to draw per call. Required
            keyword argument.
        symmetrize_self_blocks: If True, apply ``(A + A^H)/2`` to each self-cov
            block to enforce Hermitian structure numerically.

    Returns:
        Dictionary K with keys (j, k) for 0 <= k <= j < num_nodes.

    Raises:
        ValueError: on the same conditions as `cmi_dag.compute_k_blocks_multiroot`
            (root prefix, parent ordering, missing noise / root covariances),
            plus malformed edge specs, inconsistent sampler outputs, edges
            listed in ``parents`` but missing from ``edge_mats``, and
            effective-matrix shapes inconsistent with the node dimensions.
    """
    roots = sorted(roots)
    num_roots = len(roots)
    if roots != list(range(num_roots)):
        raise ValueError(
            f"roots must be the prefix {{0, ..., K-1}} in topological order, "
            f"got {roots}."
        )
    if num_roots >= num_nodes:
        raise ValueError(
            f"num_roots ({num_roots}) must be strictly less than num_nodes "
            f"({num_nodes}): the DAG must contain at least one non-root node "
            "for the channel to be non-trivial."
        )
    for r in roots:
        if r not in root_covs:
            raise ValueError(f"root_covs is missing the entry for root {r}.")

    # Pre-realize every edge once for the requested batch.
    A_eff = _realize_edges(edge_mats, batch_size)

    K: dict[tuple[int, int], torch.Tensor] = {}

    # Base case: root self-covariances (unbatched) and zero cross-covariances.
    for r in roots:
        cov = root_covs[r]
        K[(r, r)] = hermitianize(cov) if symmetrize_self_blocks else cov
    for r in roots:
        for r2 in roots:
            if r2 < r:
                d_r = K[(r, r)].shape[-1]
                d_r2 = K[(r2, r2)].shape[-1]
                K[(r, r2)] = torch.zeros(
                    d_r, d_r2, dtype=K[(r, r)].dtype, device=K[(r, r)].device
                )

    # Non-root nodes, in topological order.
    for j in range(num_roots, num_nodes):
        if j not in parents or len(parents[j]) == 0:
            raise ValueError(f"Non-root node {j} has no parents.")
        for i in parents[j]:
            if not (0 <= i < j):
                raise ValueError(
                    f"Parent {i} of node {j} violates topological order "
                    f"(0 <= i < j)."
                )
            if (j, i) not in A_eff:
                raise ValueError(
                    f"edge_mats is missing the entry for edge ({j}, {i}) "
                    f"listed in parents[{j}]."
                )
        if j not in noise_covs:
            raise ValueError(f"noise_covs is missing the entry for non-root node {j}.")
        d_j = noise_covs[j].shape[-1]
        for i in parents[j]:
            d_i = K[(i, i)].shape[-1]
            A_shape = A_eff[(j, i)].shape
            if A_shape[-2] != d_j or A_shape[-1] != d_i:
                raise ValueError(
                    f"edge ({j}, {i}): effective matrix H @ F has shape "
                    f"({A_shape[-2]}, {A_shape[-1]}) per realization, "
                    f"expected ({d_j}, {d_i}) from noise_covs[{j}] and the "
                    f"dimension of node {i}."
                )

        # (1) Cross blocks K_{jk} for k = 0, ..., j-1. Batched: (B, d_j, d_k).
        for k in range(j):
            acc: torch.Tensor | None = None
            for i in parents[j]:
                # A_eff[(j,i)]: (B, d_j, d_i); get_K(K, i, k): (B, d_i, d_k)
                # broadcasting works for either batched or unbatched K_{ik}.
                term = A_eff[(j, i)] @ get_K(K, i, k)
                acc = term if acc is None else acc + term
            assert acc is not None  # parents[j] is non-empty
            K[(j, k)] = acc

        # (2) Self block K_{jj} = sum_{i,i'} A_{j,i} K_{ii'} A_{j,i'}^H + Sigma_j.
        # Σ_j (d_j, d_j) broadcasts over the leading batch axis when added.
        acc = noise_covs[j]
        for i in parents[j]:
            A_ji = A_eff[(j, i)]
            for ip in parents[j]:
                A_jip = A_eff[(j, ip)]
                # (B, d_j, d_i) @ K_{i,i'} @ (B, d_i', d_j) -> (B, d_j, d_j).
                acc = acc + A_ji @ get_K(K, i, ip) @ A_jip.mH
        K[(j, j)] = hermitianize(acc) if symmetrize_self_blocks else acc

    # Promote any 2-D blocks (root-root, kept unbatched during the recursion
    # for memory efficiency) to the uniform 3-D layout (B, d_j, d_k) using
    # ``expand``. ``expand`` is zero-copy: the underlying root-covariance
    # tensor is shared across the batch axis via stride tricks, so the
    # promotion costs only a tensor view, not a B-fold memory blow-up.
    for key, val in list(K.items()):
        if val.dim() == 2:
            K[key] = val.unsqueeze(0).expand(batch_size, *val.shape)

    return K
