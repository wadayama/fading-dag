"""Tests for the batched K-recursion.

Coverage:
- Shape/dtype of returned K-blocks across batch sizes.
- The sampler-once-per-call invariant (each H_sampler called once).
- B=16 produces channel-distinct realizations (variance > 0 across batch).
- Error handling: malformed edge specs, missing root/noise covariances.
"""

from __future__ import annotations

import pytest
import torch

from fading_dag import samplers
from fading_dag.krecursion import compute_k_blocks_multiroot


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _two_user_mac(B: int):
    """Build a simple 2-user MAC: roots 0, 1; receiver = 2.

    Each transmitter sends through its own constant 1-edge to the receiver.
    H matrices are random per realization; F is identity (no precoding).
    """
    torch.manual_seed(0)
    d = 2
    sigma2 = 0.5
    edge_mats = {
        (2, 0): (samplers.rayleigh((d, d)), torch.eye(d, dtype=torch.complex128)),
        (2, 1): (samplers.rayleigh((d, d)), torch.eye(d, dtype=torch.complex128)),
    }
    root_covs = {0: torch.eye(d, dtype=torch.complex128), 1: torch.eye(d, dtype=torch.complex128)}
    noise_covs = {2: sigma2 * torch.eye(d, dtype=torch.complex128)}
    return d, edge_mats, root_covs, noise_covs


# --------------------------------------------------------------------------- #
# Shape / dtype                                                               #
# --------------------------------------------------------------------------- #


def test_shapes_basic_mac() -> None:
    d, edge_mats, root_covs, noise_covs = _two_user_mac(B=8)
    B = 8
    K = compute_k_blocks_multiroot(
        num_nodes=3,
        roots=[0, 1],
        parents={2: [0, 1]},
        edge_mats=edge_mats,
        root_covs=root_covs,
        noise_covs=noise_covs,
        batch_size=B,
    )
    # Canonical blocks: (0,0), (1,0), (1,1), (2,0), (2,1), (2,2).
    assert set(K.keys()) == {(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)}

    # All blocks are uniformly batched after the post-recursion promotion.
    # Root-root blocks share memory across the batch axis via ``expand``.
    assert K[(0, 0)].shape == (B, d, d)
    assert K[(1, 0)].shape == (B, d, d)
    assert K[(1, 1)].shape == (B, d, d)
    assert K[(2, 0)].shape == (B, d, d)
    assert K[(2, 1)].shape == (B, d, d)
    assert K[(2, 2)].shape == (B, d, d)

    # All blocks complex128.
    for v in K.values():
        assert v.dtype == torch.complex128


def test_root_cross_block_is_zero() -> None:
    _, edge_mats, root_covs, noise_covs = _two_user_mac(B=1)
    K = compute_k_blocks_multiroot(
        num_nodes=3,
        roots=[0, 1],
        parents={2: [0, 1]},
        edge_mats=edge_mats,
        root_covs=root_covs,
        noise_covs=noise_covs,
        batch_size=1,
    )
    assert torch.equal(K[(1, 0)], torch.zeros_like(K[(1, 0)]))


def test_self_block_is_hermitian() -> None:
    _, edge_mats, root_covs, noise_covs = _two_user_mac(B=4)
    K = compute_k_blocks_multiroot(
        num_nodes=3,
        roots=[0, 1],
        parents={2: [0, 1]},
        edge_mats=edge_mats,
        root_covs=root_covs,
        noise_covs=noise_covs,
        batch_size=4,
    )
    K22 = K[(2, 2)]
    assert torch.allclose(K22, K22.mH, atol=1e-10)


# --------------------------------------------------------------------------- #
# Sampler-once invariant                                                      #
# --------------------------------------------------------------------------- #


def test_sampler_called_exactly_once_per_edge_per_invocation() -> None:
    """The K-recursion must call each H_sampler exactly once per call.

    The cross- and self-block updates must share the realized channel, so
    counting sampler invocations is a structural check.
    """
    counts = {0: 0, 1: 0}
    d = 2

    def make_counting_sampler(edge_id: int):
        base = samplers.rayleigh((d, d))

        def sampler(B: int) -> torch.Tensor:
            counts[edge_id] += 1
            return base(B)

        return sampler

    edge_mats = {
        (2, 0): (make_counting_sampler(0), torch.eye(d, dtype=torch.complex128)),
        (2, 1): (make_counting_sampler(1), torch.eye(d, dtype=torch.complex128)),
    }
    root_covs = {0: torch.eye(d, dtype=torch.complex128), 1: torch.eye(d, dtype=torch.complex128)}
    noise_covs = {2: 0.5 * torch.eye(d, dtype=torch.complex128)}

    compute_k_blocks_multiroot(
        num_nodes=3,
        roots=[0, 1],
        parents={2: [0, 1]},
        edge_mats=edge_mats,
        root_covs=root_covs,
        noise_covs=noise_covs,
        batch_size=4,
    )

    assert counts == {0: 1, 1: 1}


# --------------------------------------------------------------------------- #
# Distinct realizations across batch                                          #
# --------------------------------------------------------------------------- #


def test_batched_realizations_are_distinct() -> None:
    """With B=16 Rayleigh samplers, the K_{22} block must vary across batch."""
    _, edge_mats, root_covs, noise_covs = _two_user_mac(B=16)
    K = compute_k_blocks_multiroot(
        num_nodes=3,
        roots=[0, 1],
        parents={2: [0, 1]},
        edge_mats=edge_mats,
        root_covs=root_covs,
        noise_covs=noise_covs,
        batch_size=16,
    )
    K22 = K[(2, 2)]
    # Variance across batch axis must be strictly positive (channel randomness).
    var_across_batch = K22.real.var(dim=0).mean().item()
    assert var_across_batch > 1e-3, f"too little variance: {var_across_batch}"


# --------------------------------------------------------------------------- #
# Consistency check against unbatched cmi-dag K-recursion (B=1 + constant)    #
# --------------------------------------------------------------------------- #


def test_b1_constant_matches_unbatched_recursion() -> None:
    """B=1 + constant samplers should match a faithful unbatched K-recursion.

    Replicates cmi-dag's algorithm inline so this test has no external dep.
    """
    torch.manual_seed(7)
    d = 2

    # Random but fixed channel matrices and precoder.
    H_20 = torch.randn(d, d, dtype=torch.complex128)
    H_21 = torch.randn(d, d, dtype=torch.complex128)
    F_20 = torch.randn(d, d, dtype=torch.complex128)
    F_21 = torch.randn(d, d, dtype=torch.complex128)

    edge_mats = {
        (2, 0): (samplers.constant(H_20), F_20),
        (2, 1): (samplers.constant(H_21), F_21),
    }
    Sigma_0 = torch.eye(d, dtype=torch.complex128)
    Sigma_1 = 0.5 * torch.eye(d, dtype=torch.complex128) + 0.1
    # Make Sigma_1 Hermitian PSD by symmetrizing.
    Sigma_1 = 0.5 * (Sigma_1 + Sigma_1.conj().T)
    Sigma_z = 0.3 * torch.eye(d, dtype=torch.complex128)

    root_covs = {0: Sigma_0, 1: Sigma_1}
    noise_covs = {2: Sigma_z}

    K_batched = compute_k_blocks_multiroot(
        num_nodes=3,
        roots=[0, 1],
        parents={2: [0, 1]},
        edge_mats=edge_mats,
        root_covs=root_covs,
        noise_covs=noise_covs,
        batch_size=1,
    )

    # Reference: unbatched algorithm with concrete A matrices.
    A_20 = H_20 @ F_20
    A_21 = H_21 @ F_21

    # Replicate cmi-dag's algorithm by hand.
    K_ref: dict[tuple[int, int], torch.Tensor] = {}
    K_ref[(0, 0)] = 0.5 * (Sigma_0 + Sigma_0.conj().T)
    K_ref[(1, 1)] = 0.5 * (Sigma_1 + Sigma_1.conj().T)
    K_ref[(1, 0)] = torch.zeros(d, d, dtype=torch.complex128)
    K_ref[(2, 0)] = A_20 @ K_ref[(0, 0)] + A_21 @ K_ref[(1, 0)]
    K_ref[(2, 1)] = A_20 @ K_ref[(1, 0)].mH + A_21 @ K_ref[(1, 1)]
    self_acc = (
        Sigma_z
        + A_20 @ K_ref[(0, 0)] @ A_20.mH
        + A_20 @ K_ref[(1, 0)].mH @ A_21.mH
        + A_21 @ K_ref[(1, 0)] @ A_20.mH
        + A_21 @ K_ref[(1, 1)] @ A_21.mH
    )
    K_ref[(2, 2)] = 0.5 * (self_acc + self_acc.mH)

    # All blocks are uniformly batched (B=1); squeeze leading axis for compare.
    for key in K_ref:
        assert torch.allclose(
            K_batched[key].squeeze(0), K_ref[key], atol=1e-10
        ), f"K{key} mismatch"


# --------------------------------------------------------------------------- #
# Error handling                                                              #
# --------------------------------------------------------------------------- #


def test_invalid_edge_spec_not_tuple_raises() -> None:
    d = 2
    edge_mats = {
        (1, 0): samplers.rayleigh((d, d)),  # missing the F factor
    }
    root_covs = {0: torch.eye(d, dtype=torch.complex128)}
    noise_covs = {1: torch.eye(d, dtype=torch.complex128)}
    with pytest.raises(ValueError, match="2-tuple"):
        compute_k_blocks_multiroot(
            num_nodes=2,
            roots=[0],
            parents={1: [0]},
            edge_mats=edge_mats,  # type: ignore[arg-type]
            root_covs=root_covs,
            noise_covs=noise_covs,
            batch_size=4,
        )


def test_invalid_edge_spec_F_not_2d_raises() -> None:
    d = 2
    edge_mats = {
        (1, 0): (samplers.rayleigh((d, d)), torch.zeros(d, dtype=torch.complex128)),  # 1-D F
    }
    root_covs = {0: torch.eye(d, dtype=torch.complex128)}
    noise_covs = {1: torch.eye(d, dtype=torch.complex128)}
    with pytest.raises(ValueError, match="2-D Tensor"):
        compute_k_blocks_multiroot(
            num_nodes=2,
            roots=[0],
            parents={1: [0]},
            edge_mats=edge_mats,
            root_covs=root_covs,
            noise_covs=noise_covs,
            batch_size=4,
        )


def test_invalid_roots_not_prefix_raises() -> None:
    d = 2
    edge_mats = {
        (2, 0): (samplers.rayleigh((d, d)), torch.eye(d, dtype=torch.complex128)),
    }
    root_covs = {0: torch.eye(d, dtype=torch.complex128), 2: torch.eye(d, dtype=torch.complex128)}
    noise_covs = {2: torch.eye(d, dtype=torch.complex128)}
    with pytest.raises(ValueError, match="prefix"):
        compute_k_blocks_multiroot(
            num_nodes=3,
            roots=[0, 2],  # not a prefix
            parents={1: [0], 2: [0]},
            edge_mats=edge_mats,
            root_covs=root_covs,
            noise_covs=noise_covs,
            batch_size=4,
        )


def test_missing_noise_cov_raises() -> None:
    d = 2
    edge_mats = {
        (1, 0): (samplers.rayleigh((d, d)), torch.eye(d, dtype=torch.complex128)),
    }
    root_covs = {0: torch.eye(d, dtype=torch.complex128)}
    noise_covs: dict[int, torch.Tensor] = {}
    with pytest.raises(ValueError, match="noise_covs is missing"):
        compute_k_blocks_multiroot(
            num_nodes=2,
            roots=[0],
            parents={1: [0]},
            edge_mats=edge_mats,
            root_covs=root_covs,
            noise_covs=noise_covs,
            batch_size=4,
        )


def test_missing_edge_for_listed_parent_raises() -> None:
    d = 2
    root_covs = {0: torch.eye(d, dtype=torch.complex128)}
    noise_covs = {1: torch.eye(d, dtype=torch.complex128)}
    with pytest.raises(ValueError, match=r"missing the entry for edge \(1, 0\)"):
        compute_k_blocks_multiroot(
            num_nodes=2,
            roots=[0],
            parents={1: [0]},
            edge_mats={},  # parents lists (1, 0) but no edge spec
            root_covs=root_covs,
            noise_covs=noise_covs,
            batch_size=4,
        )


def test_real_dtype_F_raises() -> None:
    d = 2
    edge_mats = {
        (1, 0): (samplers.rayleigh((d, d)), torch.eye(d)),  # real-valued F
    }
    root_covs = {0: torch.eye(d, dtype=torch.complex128)}
    noise_covs = {1: torch.eye(d, dtype=torch.complex128)}
    with pytest.raises(ValueError, match="complex"):
        compute_k_blocks_multiroot(
            num_nodes=2,
            roots=[0],
            parents={1: [0]},
            edge_mats=edge_mats,
            root_covs=root_covs,
            noise_covs=noise_covs,
            batch_size=4,
        )


def test_H_F_dtype_mismatch_raises() -> None:
    d = 2
    edge_mats = {
        (1, 0): (
            samplers.rayleigh((d, d), dtype=torch.complex64),
            torch.eye(d, dtype=torch.complex128),
        ),
    }
    root_covs = {0: torch.eye(d, dtype=torch.complex128)}
    noise_covs = {1: torch.eye(d, dtype=torch.complex128)}
    with pytest.raises(ValueError, match="dtype"):
        compute_k_blocks_multiroot(
            num_nodes=2,
            roots=[0],
            parents={1: [0]},
            edge_mats=edge_mats,
            root_covs=root_covs,
            noise_covs=noise_covs,
            batch_size=4,
        )


def test_edge_dimension_mismatch_raises() -> None:
    """F maps node 0 (dimension 2) through a (3, 5)-shaped factor: the
    effective matrix has 5 columns, inconsistent with d_0 = 2."""
    edge_mats = {
        (1, 0): (
            samplers.rayleigh((2, 3)),
            torch.randn(3, 5, dtype=torch.complex128),
        ),
    }
    root_covs = {0: torch.eye(2, dtype=torch.complex128)}
    noise_covs = {1: torch.eye(2, dtype=torch.complex128)}
    with pytest.raises(ValueError, match="effective matrix"):
        compute_k_blocks_multiroot(
            num_nodes=2,
            roots=[0],
            parents={1: [0]},
            edge_mats=edge_mats,
            root_covs=root_covs,
            noise_covs=noise_covs,
            batch_size=4,
        )


def test_invalid_batch_size_raises() -> None:
    d = 2
    edge_mats = {
        (1, 0): (samplers.rayleigh((d, d)), torch.eye(d, dtype=torch.complex128)),
    }
    root_covs = {0: torch.eye(d, dtype=torch.complex128)}
    noise_covs = {1: torch.eye(d, dtype=torch.complex128)}
    with pytest.raises(ValueError, match="batch_size"):
        compute_k_blocks_multiroot(
            num_nodes=2,
            roots=[0],
            parents={1: [0]},
            edge_mats=edge_mats,
            root_covs=root_covs,
            noise_covs=noise_covs,
            batch_size=0,
        )
