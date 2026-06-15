"""Unit tests for fading_dag.builder (named-node DAG builder).

Covers the acceptance criteria of builder_implementation.md (spec v0.2):
single-link / chain / multi-root (MAC) / multi-parent graphs, the round-trip
equivalence to the functional core, the structural-conformance vectors
(section 12), name/object binding, ergodic aggregation, Monte-Carlo
re-sampling, and the loud-failure requirements.

Round-trip / equality tests use the deterministic ``samplers.constant`` sampler:
a fading sampler (``rayleigh``) re-draws on every forward pass, so the builder
and a hand-built call would see different channel realizations and could not be
compared. Re-sampling itself is checked separately with ``rayleigh``.
"""

from __future__ import annotations

import pytest
import torch

from fading_dag import GaussianDAG, samplers
from fading_dag.information import conditional_mutual_information_from_k
from fading_dag.krecursion import compute_k_blocks_multiroot, get_K


DTYPE = torch.complex128


# ----------------------------- helpers -----------------------------


def _randn_complex(*shape: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    real = torch.randn(*shape, dtype=torch.float64, generator=g)
    imag = torch.randn(*shape, dtype=torch.float64, generator=g)
    return torch.complex(real, imag)


def _hermitian_psd(d: int, *, seed: int) -> torch.Tensor:
    A = _randn_complex(d, d, seed=seed)
    return A @ A.mH + torch.eye(d, dtype=DTYPE)


def _const_edge(d: int, *, seed: int):
    """A deterministic EdgeSpec (constant sampler, identity factor F)."""
    H = _randn_complex(d, d, seed=seed)
    return (samplers.constant(H), torch.eye(d, dtype=DTYPE))


# ============================================================
# single-link X -> Y : round-trip vs the functional core (deterministic)
# ============================================================


def test_single_link_cmi_roundtrip():
    d, B = 2, 8
    Sx = _hermitian_psd(d, seed=100)
    Sz = _hermitian_psd(d, seed=200)
    edge = _const_edge(d, seed=1)

    dag = GaussianDAG()
    dag.add_source("X", cov=Sx)
    dag.add_node("Y", parents={"X": edge}, noise=Sz)
    mi_builder = dag.cmi(A=["X"], B=["Y"], batch_size=B)

    K = compute_k_blocks_multiroot(
        num_nodes=2,
        roots=[0],
        parents={1: [0]},
        edge_mats={(1, 0): edge},
        root_covs={0: Sx},
        noise_covs={1: Sz},
        batch_size=B,
    )
    mi_core = conditional_mutual_information_from_k(K, A=[0], B=[1])

    assert mi_builder.shape == (B,)
    assert torch.allclose(mi_builder, mi_core, atol=1e-10)


# ============================================================
# 2-user MAC round-trip (multiroot + conditional, deterministic)
# ============================================================


def test_mac_cmi_roundtrip():
    d, B = 2, 8
    S1 = _hermitian_psd(d, seed=100)
    S2 = _hermitian_psd(d, seed=101)
    e0 = _const_edge(d, seed=1)
    e1 = _const_edge(d, seed=2)
    N_Y = _hermitian_psd(d, seed=200)

    dag = GaussianDAG()
    dag.add_source("X1", cov=S1)
    dag.add_source("X2", cov=S2)
    dag.add_node("Y", parents={"X1": e0, "X2": e1}, noise=N_Y)

    K = compute_k_blocks_multiroot(
        num_nodes=3,
        roots=[0, 1],
        parents={2: [0, 1]},
        edge_mats={(2, 0): e0, (2, 1): e1},
        root_covs={0: S1, 1: S2},
        noise_covs={2: N_Y},
        batch_size=B,
    )
    assert torch.allclose(
        dag.cmi(A=["X1"], B=["Y"], C=["X2"], batch_size=B),
        conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1]),
        atol=1e-10,
    )
    assert torch.allclose(
        dag.cmi(A=["X1", "X2"], B=["Y"], batch_size=B),
        conditional_mutual_information_from_k(K, A=[0, 1], B=[2], C=[]),
        atol=1e-10,
    )


# ============================================================
# chain X -> Y -> Z : round-trip (deterministic)
# ============================================================


def test_chain_cmi_roundtrip():
    d, B = 2, 8
    Sx = _hermitian_psd(d, seed=100)
    e_xy = _const_edge(d, seed=1)
    e_yz = _const_edge(d, seed=2)
    N_Y = _hermitian_psd(d, seed=201)
    N_Z = _hermitian_psd(d, seed=202)

    dag = GaussianDAG()
    dag.add_source("X", cov=Sx)
    dag.add_node("Y", parents={"X": e_xy}, noise=N_Y)
    dag.add_node("Z", parents={"Y": e_yz}, noise=N_Z)

    K = compute_k_blocks_multiroot(
        num_nodes=3,
        roots=[0],
        parents={1: [0], 2: [1]},
        edge_mats={(1, 0): e_xy, (2, 1): e_yz},
        root_covs={0: Sx},
        noise_covs={1: N_Y, 2: N_Z},
        batch_size=B,
    )
    assert torch.allclose(
        dag.cmi(A=["X"], B=["Z"], C=["Y"], batch_size=B),
        conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1]),
        atol=1e-10,
    )
    assert torch.allclose(dag.cov("Z", batch_size=B), get_K(K, 2, 2), atol=1e-10)


# ============================================================
# ergodic_capacity == per-realization mean (deterministic)
# ============================================================


def test_ergodic_capacity_equals_mean():
    d, B = 2, 8
    Sx = _hermitian_psd(d, seed=100)
    Sz = _hermitian_psd(d, seed=200)
    edge = _const_edge(d, seed=1)

    dag = GaussianDAG()
    dag.add_source("X", cov=Sx)
    dag.add_node("Y", parents={"X": edge}, noise=Sz)

    assert torch.allclose(
        dag.ergodic_capacity(A=["X"], B=["Y"], batch_size=B),
        dag.cmi(A=["X"], B=["Y"], batch_size=B).mean(),
        atol=1e-10,
    )


# ============================================================
# Structural-conformance vectors (spec section 12, structure only)
# ============================================================


def test_structure_chain():
    dag = GaussianDAG()
    dag.add_source("X", cov="Sx")
    dag.add_node("Y", parents={"X": "e_XY"}, noise="N_Y")
    dag.add_node("Z", parents={"Y": "e_YZ"}, noise="N_Z")
    order, sources, parents, edges = dag._lower_structure()
    assert order == ["X", "Y", "Z"]
    assert sources == {0}
    assert parents == {1: [0], 2: [1]}
    assert edges == {(1, 0), (2, 1)}


def test_structure_two_source_mac():
    dag = GaussianDAG()
    dag.add_source("X", cov="Sx")
    dag.add_source("Y", cov="Sy")
    dag.add_node("Z", parents={"X": "e_XZ", "Y": "e_YZ"}, noise="N_Z")
    order, sources, parents, edges = dag._lower_structure()
    assert order == ["X", "Y", "Z"]
    assert sources == {0, 1}
    assert parents == {2: [0, 1]}
    assert edges == {(2, 0), (2, 1)}


def test_structure_diamond():
    dag = GaussianDAG()
    dag.add_source("X", cov="Sx")
    dag.add_node("Y", parents={"X": "e_XY"}, noise="N_Y")
    dag.add_node("W", parents={"X": "e_XW"}, noise="N_W")
    dag.add_node("Z", parents={"Y": "e_YZ", "W": "e_WZ"}, noise="N_Z")
    order, sources, parents, edges = dag._lower_structure()
    assert order == ["X", "Y", "W", "Z"]
    assert sources == {0}
    assert parents == {1: [0], 2: [0], 3: [1, 2]}
    assert edges == {(1, 0), (2, 0), (3, 1), (3, 2)}


# ============================================================
# Binding: name-or-object resolved at query time (spec section 8)
# ============================================================


def test_bind_by_name_matches_concrete():
    d, B = 2, 8
    Sx = _hermitian_psd(d, seed=100)
    Sz = _hermitian_psd(d, seed=200)
    edge = _const_edge(d, seed=1)
    binding = {"Sx": Sx, "Sz": Sz, "e_XY": edge}

    by_name = GaussianDAG()
    by_name.add_source("X", cov="Sx")
    by_name.add_node("Y", parents={"X": "e_XY"}, noise="Sz")
    mi_named = by_name.cmi(A=["X"], B=["Y"], batch_size=B, bind=binding)

    by_obj = GaussianDAG()
    by_obj.add_source("X", cov=Sx)
    by_obj.add_node("Y", parents={"X": edge}, noise=Sz)
    mi_obj = by_obj.cmi(A=["X"], B=["Y"], batch_size=B)

    assert torch.allclose(mi_named, mi_obj, atol=1e-12)


def test_unbound_name_raises():
    dag = GaussianDAG()
    dag.add_source("X", cov="Sx")
    dag.add_node("Y", parents={"X": "e_XY"}, noise="N_Y")
    with pytest.raises(ValueError, match="not bound"):
        dag.cmi(A=["X"], B=["Y"], batch_size=4,
                bind={"Sx": _hermitian_psd(2, seed=1)})


# ============================================================
# Monte-Carlo re-sampling: rayleigh re-draws on every query
# ============================================================


def test_rayleigh_resamples_each_call():
    d, B = 2, 64
    torch.manual_seed(0)
    F = torch.eye(d, dtype=DTYPE)
    dag = GaussianDAG()
    dag.add_source("X", cov=torch.eye(d, dtype=DTYPE))
    dag.add_node(
        "Y",
        parents={"X": (samplers.rayleigh((d, d)), F)},
        noise=0.5 * torch.eye(d, dtype=DTYPE),
    )
    I1 = dag.cmi(A=["X"], B=["Y"], batch_size=B)
    I2 = dag.cmi(A=["X"], B=["Y"], batch_size=B)
    assert I1.shape == (B,)
    assert torch.isfinite(I1).all()
    # Two independent mini-batches -> different draws.
    assert not torch.allclose(I1, I2)


# ============================================================
# Differentiability survives the builder (core autograd preserved)
# ============================================================


def test_ergodic_capacity_is_differentiable():
    d, B = 2, 16
    torch.manual_seed(0)
    F = (0.2 * _randn_complex(d, d, seed=7)).requires_grad_(True)
    dag = GaussianDAG()
    dag.add_source("X", cov=torch.eye(d, dtype=DTYPE))
    dag.add_node(
        "Y",
        parents={"X": (samplers.rayleigh((d, d)), F)},
        noise=0.5 * torch.eye(d, dtype=DTYPE),
    )
    ce = dag.ergodic_capacity(A=["X"], B=["Y"], batch_size=B)
    ce.backward()
    assert F.grad is not None
    assert torch.isfinite(F.grad).all()


# ============================================================
# Loud failures for unsupported / invalid constructs (spec section 4.4)
# ============================================================


def test_unknown_parent_rejected():
    dag = GaussianDAG()
    dag.add_source("X", cov="Sx")
    with pytest.raises(ValueError, match="Unknown parent"):
        dag.add_node("Z", parents={"Q": "e"}, noise="N_Z")


def test_duplicate_name_rejected():
    dag = GaussianDAG()
    dag.add_source("X", cov="Sx")
    with pytest.raises(ValueError, match="Duplicate"):
        dag.add_source("X", cov="Sx2")


def test_parentless_node_rejected():
    dag = GaussianDAG()
    dag.add_source("X", cov="Sx")
    with pytest.raises(ValueError, match="no parents"):
        dag.add_node("Y", parents={}, noise="N_Y")


def test_self_loop_rejected():
    dag = GaussianDAG()
    dag.add_source("X", cov="Sx")
    with pytest.raises(ValueError, match="Unknown parent"):
        dag.add_node("Y", parents={"Y": "e"}, noise="N_Y")


def test_cmi_overlapping_sets_rejected():
    d, B = 2, 4
    dag = GaussianDAG()
    dag.add_source("X1", cov=_hermitian_psd(d, seed=100))
    dag.add_source("X2", cov=_hermitian_psd(d, seed=101))
    dag.add_node("Y", parents={"X1": _const_edge(d, seed=1),
                               "X2": _const_edge(d, seed=2)},
                 noise=_hermitian_psd(d, seed=200))
    with pytest.raises(ValueError, match="disjoint"):
        dag.cmi(A=["X1"], B=["X1"], C=["X2"], batch_size=B)


def test_sources_only_query_rejected():
    d, B = 2, 4
    dag = GaussianDAG()
    dag.add_source("X1", cov=_hermitian_psd(d, seed=100))
    dag.add_source("X2", cov=_hermitian_psd(d, seed=101))
    with pytest.raises(ValueError, match="num_roots"):
        dag.cmi(A=["X1"], B=["X2"], batch_size=B)
