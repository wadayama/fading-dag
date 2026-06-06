"""Tests for outage probability helpers."""

from __future__ import annotations

import pytest
import torch

from fading_dag.outage import (
    ergodic_capacity,
    outage_probability,
    outage_probability_smooth,
)


# --------------------------------------------------------------------------- #
# ergodic_capacity                                                            #
# --------------------------------------------------------------------------- #


def test_ergodic_capacity_matches_mean() -> None:
    I = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = ergodic_capacity(I)
    assert torch.isclose(out, torch.tensor(2.5))


def test_ergodic_capacity_differentiable() -> None:
    I = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    out = ergodic_capacity(I)
    out.backward()
    assert torch.allclose(I.grad, torch.full_like(I, 1 / 4))


def test_ergodic_capacity_empty_raises() -> None:
    with pytest.raises(ValueError):
        ergodic_capacity(torch.tensor([]))


# --------------------------------------------------------------------------- #
# outage_probability (raw indicator)                                          #
# --------------------------------------------------------------------------- #


def test_outage_probability_exact_count() -> None:
    I = torch.tensor([0.0, 1.0, 2.0, 3.0])
    out = outage_probability(I, R=1.5)
    # Below 1.5: {0.0, 1.0} -> 2 / 4 = 0.5.
    assert torch.isclose(out, torch.tensor(0.5))


def test_outage_probability_all_above_threshold() -> None:
    I = torch.tensor([10.0, 20.0, 30.0])
    out = outage_probability(I, R=5.0)
    assert torch.isclose(out, torch.tensor(0.0))


def test_outage_probability_all_below_threshold() -> None:
    I = torch.tensor([0.0, 1.0, 2.0])
    out = outage_probability(I, R=10.0)
    assert torch.isclose(out, torch.tensor(1.0))


def test_outage_probability_does_not_propagate_grad() -> None:
    I = torch.tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    out = outage_probability(I, R=1.5)
    # The raw indicator detaches; the output should not require grad.
    assert not out.requires_grad


def test_outage_probability_empty_raises() -> None:
    with pytest.raises(ValueError):
        outage_probability(torch.tensor([]), R=1.0)


# --------------------------------------------------------------------------- #
# outage_probability_smooth (sigmoid surrogate)                               #
# --------------------------------------------------------------------------- #


def test_outage_smooth_differentiable() -> None:
    I = torch.tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    out = outage_probability_smooth(I, R=1.5, tau=0.5)
    assert out.requires_grad
    out.backward()
    assert I.grad is not None
    # All gradient entries should be negative: increasing I decreases outage.
    assert (I.grad <= 0).all()


def test_outage_smooth_converges_to_hard_as_tau_decreases() -> None:
    """Outage_smooth(I, R, tau) -> outage_probability(I, R) as tau -> 0."""
    torch.manual_seed(0)
    I = torch.randn(2000)
    R = 0.3
    hard = outage_probability(I, R=R).item()
    for tau in (1.0, 0.5, 0.1, 0.02):
        smooth = outage_probability_smooth(I, R=R, tau=tau).item()
        if tau == 1.0:
            tau1 = abs(smooth - hard)
        if tau == 0.02:
            tau_small = abs(smooth - hard)
    assert tau_small < tau1, "smooth -> hard convergence not visible"
    assert tau_small < 0.02, f"residual gap at small tau too large: {tau_small}"


def test_outage_smooth_invalid_tau_raises() -> None:
    I = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="tau"):
        outage_probability_smooth(I, R=1.5, tau=0.0)
    with pytest.raises(ValueError, match="tau"):
        outage_probability_smooth(I, R=1.5, tau=-0.5)


def test_outage_smooth_in_unit_interval() -> None:
    torch.manual_seed(1)
    I = torch.randn(256)
    out = outage_probability_smooth(I, R=0.0, tau=0.1)
    assert 0.0 < out.item() < 1.0
