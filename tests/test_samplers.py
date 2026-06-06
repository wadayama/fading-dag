"""Unit tests for fading_dag.samplers."""

from __future__ import annotations

import pytest
import torch

from fading_dag import samplers


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _seeded_generator(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------------- #
# rayleigh                                                                    #
# --------------------------------------------------------------------------- #


def test_rayleigh_shape_and_dtype() -> None:
    sampler = samplers.rayleigh((4, 2))
    H = sampler(8)
    assert H.shape == (8, 4, 2)
    assert H.dtype == torch.complex128
    # Per-entry variance (each entry CN(0,1)) should be ~1 on average.
    var = (H.abs() ** 2).mean().item()
    assert 0.5 < var < 1.5, f"unit-variance check failed: var={var}"


def test_rayleigh_complex64() -> None:
    sampler = samplers.rayleigh((3, 3), dtype=torch.complex64)
    H = sampler(5)
    assert H.dtype == torch.complex64


def test_rayleigh_invalid_shape_raises() -> None:
    with pytest.raises(ValueError):
        samplers.rayleigh((4,))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        samplers.rayleigh((0, 3))


def test_rayleigh_independent_draws() -> None:
    """Two calls must produce different samples (PyTorch global RNG)."""
    torch.manual_seed(0)
    sampler = samplers.rayleigh((4, 2))
    H1 = sampler(8)
    H2 = sampler(8)
    assert not torch.allclose(H1, H2)


# --------------------------------------------------------------------------- #
# ricean                                                                      #
# --------------------------------------------------------------------------- #


def test_ricean_K_zero_recovers_rayleigh_variance() -> None:
    """K=0 collapses Ricean to Rayleigh: per-entry variance ~ 1."""
    torch.manual_seed(0)
    H_LOS = torch.zeros(3, 2, dtype=torch.complex128)
    sampler = samplers.ricean((3, 2), H_LOS, K=0.0)
    H = sampler(2000)
    var = (H.abs() ** 2).mean().item()
    assert 0.85 < var < 1.15, f"K=0 variance check: var={var}"


def test_ricean_K_large_concentrates_at_LOS() -> None:
    """K very large -> samples concentrate near H_LOS."""
    torch.manual_seed(0)
    H_LOS = torch.full((2, 2), 1.0 + 0.0j, dtype=torch.complex128)
    sampler = samplers.ricean((2, 2), H_LOS, K=1000.0)
    H = sampler(64)
    # Each sample's mean magnitude should be close to |H_LOS_entry| * sqrt(K/(K+1)) ~ 1.
    mean_mag = H.mean(dim=0).abs()
    target = (1000.0 / 1001.0) ** 0.5
    assert torch.allclose(mean_mag, torch.full_like(mean_mag, target), atol=0.05)


def test_ricean_shape_mismatch_raises() -> None:
    H_LOS = torch.zeros(2, 2, dtype=torch.complex128)
    with pytest.raises(ValueError):
        samplers.ricean((3, 3), H_LOS, K=1.0)


def test_ricean_negative_K_raises() -> None:
    H_LOS = torch.zeros(2, 2, dtype=torch.complex128)
    with pytest.raises(ValueError):
        samplers.ricean((2, 2), H_LOS, K=-0.1)


# --------------------------------------------------------------------------- #
# kronecker                                                                   #
# --------------------------------------------------------------------------- #


def test_kronecker_identity_correlations_match_rayleigh() -> None:
    """R_rx = R_tx = I should reduce kronecker to plain Rayleigh."""
    torch.manual_seed(0)
    d_out, d_in = 3, 2
    R_rx = torch.eye(d_out, dtype=torch.complex128)
    R_tx = torch.eye(d_in, dtype=torch.complex128)
    sampler = samplers.kronecker((d_out, d_in), R_rx, R_tx)
    H = sampler(2000)
    var = (H.abs() ** 2).mean().item()
    assert 0.85 < var < 1.15


def test_kronecker_rx_correlation_visible() -> None:
    """With R_rx nontrivial, row-row correlation should be non-zero."""
    torch.manual_seed(0)
    d_out, d_in = 2, 4
    # Strong correlation between rows 0 and 1.
    R_rx = torch.tensor(
        [[1.0 + 0j, 0.9 + 0j], [0.9 + 0j, 1.0 + 0j]], dtype=torch.complex128
    )
    R_tx = torch.eye(d_in, dtype=torch.complex128)
    sampler = samplers.kronecker((d_out, d_in), R_rx, R_tx)
    H = sampler(4000)
    # Empirical cross-correlation of row 0 and row 1.
    row0 = H[:, 0, :]
    row1 = H[:, 1, :]
    cross = (row0.conj() * row1).real.mean().item()
    # Should be near the true correlation value (averaged over d_in columns
    # and B batches gives one number); analytical expectation ~ 0.9.
    assert 0.7 < cross < 1.1, f"correlation visibility: cross={cross}"


def test_kronecker_shape_mismatch_raises() -> None:
    R_rx = torch.eye(3, dtype=torch.complex128)
    R_tx = torch.eye(2, dtype=torch.complex128)
    with pytest.raises(ValueError):
        samplers.kronecker((2, 2), R_rx, R_tx)


# --------------------------------------------------------------------------- #
# scaled_rayleigh                                                             #
# --------------------------------------------------------------------------- #


def test_scaled_rayleigh_scalar_sigma() -> None:
    torch.manual_seed(0)
    sigma = 2.0
    sampler = samplers.scaled_rayleigh((3, 3), sigma)
    H = sampler(2000)
    var = (H.abs() ** 2).mean().item()
    # Variance should be sigma^2 = 4.
    assert 3.5 < var < 4.5


def test_scaled_rayleigh_tensor_sigma_broadcasts() -> None:
    torch.manual_seed(0)
    sigma = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    sampler = samplers.scaled_rayleigh((2, 2), sigma)
    H = sampler(4000)
    # Per-entry mean magnitude squared.
    var = (H.abs() ** 2).mean(dim=0)
    expected = sigma**2
    assert torch.allclose(var.double(), expected.double(), atol=0.4)


def test_scaled_rayleigh_negative_sigma_raises() -> None:
    with pytest.raises(ValueError):
        samplers.scaled_rayleigh((2, 2), -1.0)
    with pytest.raises(ValueError):
        samplers.scaled_rayleigh(
            (2, 2), torch.tensor([[1.0, -0.5], [1.0, 1.0]])
        )


# --------------------------------------------------------------------------- #
# constant                                                                    #
# --------------------------------------------------------------------------- #


def test_constant_emits_same_matrix_each_draw() -> None:
    H_fixed = torch.tensor(
        [[1.0 + 0j, 2.0 + 0j], [3.0 + 0j, 4.0 + 0j]], dtype=torch.complex128
    )
    sampler = samplers.constant(H_fixed)
    H = sampler(5)
    assert H.shape == (5, 2, 2)
    for b in range(5):
        assert torch.equal(H[b], H_fixed)


def test_constant_non_complex_raises() -> None:
    H_real = torch.eye(3)
    with pytest.raises(ValueError):
        samplers.constant(H_real)


def test_constant_non_2d_raises() -> None:
    H_3d = torch.zeros(2, 2, 2, dtype=torch.complex128)
    with pytest.raises(ValueError):
        samplers.constant(H_3d)
