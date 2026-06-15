"""Named-node DAG builder: a declarative front-end over the fading core.

This module is a *pure, backward-compatible addition* to the library. It adds
no new behavior to the numerical core; it only provides a convenience surface
that *lowers* a named-node DAG declaration to the existing functional API
(``compute_k_blocks_multiroot`` -> ``conditional_mutual_information_from_k`` /
``ergodic_capacity`` / ``get_K``).

Worked example (an ergodic single-link MIMO channel ``X -> Y``)::

    dag = GaussianDAG()
    dag.add_source("X", cov=Sigma_X)
    dag.add_node("Y", parents={"X": (samplers.rayleigh((d, d)), F)}, noise=Sigma_Z)

    I  = dag.cmi(A=["X"], B=["Y"], batch_size=256)              # per-realization (B,)
    Ce = dag.ergodic_capacity(A=["X"], B=["Y"], batch_size=256) # mini-batch mean
    Sigma_Y = dag.cov("Y", batch_size=256)                      # batched (B, d, d)

Multiple sources (e.g. a fading MAC) are declared by calling ``add_source`` more
than once::

    dag.add_source("X1", cov=S1)
    dag.add_source("X2", cov=S2)
    dag.add_node("Y", parents={"X1": (samplers.rayleigh((d, d)), F1),
                               "X2": (samplers.rayleigh((d, d)), F2)}, noise=N_Y)
    I1 = dag.cmi(A=["X1"], B=["Y"], C=["X2"], batch_size=256)

Profiles (see builder_implementation.md, spec v0.2): this library implements the
*conditional* (``cmi(A, B, C)``), *multiroot* (more than one source), and
*stochastic/batch* profiles. Per spec section 7 the *structural* builder is
unchanged from the deterministic siblings; the fading nature lives entirely in
the binding/shape layer:

* each edge value is an ``EdgeSpec = (H_sampler, F)`` 2-tuple rather than a plain
  matrix (the sampler draws a fresh mini-batch of channel realizations on every
  forward pass; ``F`` is the deterministic, shared, controllable factor);
* every query takes a required ``batch_size`` keyword and re-runs the core (and
  therefore re-samples) on each call, matching the SGD closure convention;
* quantities are per-realization tensors of shape ``(B,)`` (``cmi``) or their
  mini-batch mean (``ergodic_capacity``).

The core has no correlated-roots support (``cross_root_covs``), so unlike the
``cmi-dag`` builder there is no ``add_root_correlation`` method here.

Matrices/edges may be given either as concrete objects (a tensor, or an
``EdgeSpec`` tuple) used as-is, or by name (strings resolved at query time via a
``bind={name: object}`` mapping). An unbound name raises ``ValueError`` -- data
is never fabricated (spec section 8).

Conventions inherited from the core (``fading_dag.krecursion``): node indices
are 0-based; roots are the prefix ``{0, ..., K-1}`` in topological order; edge
keys are ``(j, i)`` for the edge ``i -> j`` with ``i < j``. Canonical node
indices are assigned by a stable topological sort (Kahn's algorithm, FIFO queue,
ties broken by build/call order); because only sources have in-degree 0 they
always receive the contiguous prefix ``0, ..., K-1`` -- exactly the root
convention the core requires.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence, Union

import torch

from fading_dag.information import conditional_mutual_information_from_k
from fading_dag.krecursion import compute_k_blocks_multiroot, get_K
from fading_dag.outage import ergodic_capacity as _ergodic_capacity

# A build-time reference: a concrete object (a covariance tensor, or an
# ``EdgeSpec = (H_sampler, F)`` tuple) used as-is, or a name (string) resolved at
# query time via the ``bind`` mapping.
MatrixRef = Union[str, torch.Tensor, tuple]


class GaussianDAG:
    """Declarative named-node builder for a fading multi-root linear Gaussian DAG.

    See the module docstring for the worked example, the supported profiles
    (conditional / multiroot / stochastic-batch), and the binding rules. The
    builder is a thin layer: it records structure and matrix/edge references at
    build time, then lowers to the library's functional core when a query
    (``cmi`` / ``ergodic_capacity`` / ``cov``) runs.
    """

    def __init__(self) -> None:
        # name -> root covariance reference (unbatched tensor or name string).
        self._sources: dict[str, MatrixRef] = {}
        # name -> (parents: {parent_name: edge_ref}, noise_ref).
        # An edge_ref is an EdgeSpec (H_sampler, F) tuple or a name string.
        self._nodes: dict[str, tuple[dict[str, MatrixRef], MatrixRef]] = {}
        # Build/call order; used to derive canonical indices (spec section 12).
        self._order: list[str] = []

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_source(self, name: str, *, cov: MatrixRef) -> "GaussianDAG":
        """Declare a source (root) node with covariance ``cov``.

        This library is multi-root: any number of sources may be declared; the
        sources are mutually independent (the fading core does not model
        correlated roots).

        Returns ``self`` to allow chaining.
        """
        self._check_new(name)
        self._sources[name] = cov
        self._order.append(name)
        return self

    def add_node(
        self,
        name: str,
        *,
        parents: dict[str, MatrixRef],
        noise: MatrixRef,
    ) -> "GaussianDAG":
        """Declare a non-source node from its ``parents`` and own ``noise``.

        ``parents`` maps each parent's name to the edge on that link. An edge is
        an ``EdgeSpec = (H_sampler, F)`` 2-tuple (or a name resolved at query
        time): ``H_sampler(batch_size)`` returns a ``(B, d_j, d_mid)`` batch of
        channel realizations and ``F`` is the shared deterministic factor.
        Every parent must already be declared (this enforces acyclicity and a
        valid topological order, and catches self-loops). ``parents`` must be
        non-empty -- a parentless node is a source; use ``add_source``.

        Returns ``self`` to allow chaining.
        """
        self._check_new(name)
        if not parents:
            raise ValueError(
                f"Node {name!r} has no parents. A parentless node is a source; "
                "use add_source(name, cov=...)."
            )
        for p in parents:
            if p not in self._sources and p not in self._nodes:
                raise ValueError(
                    f"Unknown parent {p!r} of node {name!r}: declare it with "
                    "add_source/add_node before referencing it."
                )
        self._nodes[name] = (dict(parents), noise)
        self._order.append(name)
        return self

    def _check_new(self, name: str) -> None:
        if name in self._sources or name in self._nodes:
            raise ValueError(f"Duplicate node {name!r}.")

    # ------------------------------------------------------------------
    # Lowering: names -> canonical indices / core inputs
    # ------------------------------------------------------------------

    def _canonical_index(self) -> dict[str, int]:
        """Assign canonical 0-based indices via a stable topological sort.

        Kahn's algorithm with a FIFO queue; the queue is seeded, and every tie
        broken, by build/call order (``self._order``). Deterministic for a
        given build script (spec section 12). Only sources have in-degree 0, so
        they are enqueued first (in call order) and receive the contiguous
        prefix ``0, ..., K-1`` -- matching the core's "roots are the prefix
        {0, ..., K-1}" requirement.
        """
        children: dict[str, list[str]] = {n: [] for n in self._order}
        indeg: dict[str, int] = {n: 0 for n in self._order}
        for name, (parents, _) in self._nodes.items():
            for p in parents:
                children[p].append(name)
                indeg[name] += 1

        queue: deque[str] = deque(n for n in self._order if indeg[n] == 0)
        index: dict[str, int] = {}
        nxt = 0
        while queue:
            n = queue.popleft()
            index[n] = nxt
            nxt += 1
            for c in children[n]:  # children already in build order
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)

        if len(index) != len(self._order):
            # Unreachable given add_node's pre-declared-parent rule, but guard
            # against a cycle rather than silently dropping nodes.
            raise ValueError("DAG contains a cycle; cannot order nodes.")
        return index

    def _lower_structure(
        self,
    ) -> tuple[list[str], set[int], dict[int, list[int]], set[tuple[int, int]]]:
        """Return the index-based structure (no matrix/edge resolution).

        Yields ``(order, sources, parents, edges)`` -- node names by canonical
        index, the set of source/root indices, the parent-index lists, and the
        ``(child, parent)`` index pairs. Used by the structural-conformance
        checks (spec section 12).
        """
        idx = self._canonical_index()
        order = [name for name, _ in sorted(idx.items(), key=lambda kv: kv[1])]
        sources = {idx[s] for s in self._sources}
        parents = {
            idx[n]: [idx[p] for p in ps] for n, (ps, _) in self._nodes.items()
        }
        edges = {
            (idx[n], idx[p])
            for n, (ps, _) in self._nodes.items()
            for p in ps
        }
        return order, sources, parents, edges

    @staticmethod
    def _resolve(m: MatrixRef, bind: dict[str, object] | None) -> object:
        """Resolve a build-time reference to a concrete object.

        A concrete object (a tensor, or an ``EdgeSpec`` tuple) is used as-is; a
        name (string) is looked up in ``bind``. An unbound name raises
        ``ValueError`` -- never fabricated.
        """
        if isinstance(m, str):
            if bind is None or m not in bind:
                raise ValueError(
                    f"Reference {m!r} is not bound. Pass it via "
                    "bind={...} on the query."
                )
            return bind[m]
        return m

    def _lower_core_inputs(self, bind: dict[str, object] | None) -> dict:
        """Build the keyword arguments consumed by
        ``compute_k_blocks_multiroot`` (except ``batch_size``), resolving every
        covariance and edge reference."""
        if not self._sources:
            raise ValueError("No source declared; call add_source(...) first.")
        idx = self._canonical_index()

        roots = sorted(idx[s] for s in self._sources)
        parents = {
            idx[n]: [idx[p] for p in ps] for n, (ps, _) in self._nodes.items()
        }
        edge_mats = {
            (idx[n], idx[p]): self._resolve(e, bind)
            for n, (ps, _) in self._nodes.items()
            for p, e in ps.items()
        }
        root_covs = {
            idx[s]: self._resolve(cv, bind) for s, cv in self._sources.items()
        }
        noise_covs = {
            idx[n]: self._resolve(nz, bind) for n, (_, nz) in self._nodes.items()
        }
        return dict(
            num_nodes=len(self._order),
            roots=roots,
            parents=parents,
            edge_mats=edge_mats,
            root_covs=root_covs,
            noise_covs=noise_covs,
        )

    def _require_known(self, name: str) -> None:
        if name not in self._sources and name not in self._nodes:
            raise ValueError(f"Unknown node {name!r}.")

    def _compute_k(
        self, bind: dict[str, object] | None, batch_size: int
    ) -> dict[tuple[int, int], torch.Tensor]:
        """Lower and run one batched forward pass (re-samples every edge)."""
        return compute_k_blocks_multiroot(
            **self._lower_core_inputs(bind), batch_size=batch_size
        )

    # ------------------------------------------------------------------
    # Queries (each lowers to the core and returns its result)
    # ------------------------------------------------------------------

    def cmi(
        self,
        A: Sequence[str],
        B: Sequence[str],
        C: Sequence[str] = (),
        *,
        batch_size: int,
        bind: dict[str, object] | None = None,
        jitter: float = 0.0,
    ) -> torch.Tensor:
        """Per-realization conditional MI ``I(V_A; V_B | V_C)``, shape ``(B,)``.

        ``A``, ``B``, ``C`` are lists of node names (``C`` optional). The query
        draws a fresh mini-batch of ``batch_size`` channel realizations and
        returns the per-realization MI in nats -- exactly what
        ``conditional_mutual_information_from_k`` returns. Non-empty /
        pairwise-disjoint requirements are enforced by the core.
        """
        for nm in (*A, *B, *C):
            self._require_known(nm)
        idx = self._canonical_index()
        K = self._compute_k(bind, batch_size)
        return conditional_mutual_information_from_k(
            K,
            A=[idx[n] for n in A],
            B=[idx[n] for n in B],
            C=[idx[n] for n in C],
            jitter=jitter,
        )

    def ergodic_capacity(
        self,
        A: Sequence[str],
        B: Sequence[str],
        C: Sequence[str] = (),
        *,
        batch_size: int,
        bind: dict[str, object] | None = None,
        jitter: float = 0.0,
    ) -> torch.Tensor:
        """Ergodic conditional MI: the mini-batch mean of ``cmi`` (a scalar).

        Differentiable through the controllable factors, so it can be used
        directly as an SGD-ascent objective.
        """
        I_samples = self.cmi(
            A, B, C, batch_size=batch_size, bind=bind, jitter=jitter
        )
        return _ergodic_capacity(I_samples)

    def cov(
        self,
        node: str,
        *,
        batch_size: int,
        bind: dict[str, object] | None = None,
    ) -> torch.Tensor:
        """Batched self-covariance block ``Sigma_node = K_{node,node}``.

        Lowers to ``compute_k_blocks_multiroot`` and returns the canonical
        self-block via ``get_K`` (shape ``(B, d, d)``).
        """
        self._require_known(node)
        idx = self._canonical_index()
        K = self._compute_k(bind, batch_size)
        return get_K(K, idx[node], idx[node])
