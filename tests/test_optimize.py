"""Tests for sgd_ascent / sgd_descent on simple Rayleigh problems."""

from __future__ import annotations

import pytest
import torch

from fading_dag import samplers
from fading_dag.information import conditional_mutual_information_from_k
from fading_dag.krecursion import compute_k_blocks_multiroot
from fading_dag.optimize import sgd_ascent, sgd_descent
from fading_dag.outage import outage_probability_smooth
from fading_dag.projections import project_frobenius_ball


# --------------------------------------------------------------------------- #
# 2x2 Rayleigh MIMO: maximize E[I(X; HFX + Z)] via sgd_ascent.                #
# --------------------------------------------------------------------------- #


def _make_mimo_closure(F: torch.Tensor, batch_size: int):
    d = F.shape[0]
    sigma2 = 0.5
    H_sampler = samplers.rayleigh((d, d))
    sigma_x = torch.eye(d, dtype=torch.complex128)
    sigma_z = sigma2 * torch.eye(d, dtype=torch.complex128)

    def closure() -> torch.Tensor:
        K = compute_k_blocks_multiroot(
            num_nodes=2,
            roots=[0],
            parents={1: [0]},
            edge_mats={(1, 0): (H_sampler, F)},
            root_covs={0: sigma_x},
            noise_covs={1: sigma_z},
            batch_size=batch_size,
        )
        I = conditional_mutual_information_from_k(K, A=[0], B=[1])
        return I.real.mean()

    return closure


def test_sgd_ascent_increases_rayleigh_mimo_ergodic_capacity() -> None:
    """sgd_ascent must drive E[I] upward; trailing-50 window mean must increase."""
    torch.manual_seed(123)
    d = 2
    P = 4.0
    # Initialise F at a small magnitude well inside the Frobenius ball.
    F = (0.1 * torch.randn(d, d, dtype=torch.complex128)).requires_grad_(True)
    closure = _make_mimo_closure(F, batch_size=32)

    def projector(params: list[torch.Tensor]) -> list[torch.Tensor]:
        return [project_frobenius_ball(p, P=P) for p in params]

    history = sgd_ascent(
        closure,
        [F],
        step_size=0.05,
        num_iters=500,
        projector=projector,
    )

    early = sum(history[:50]) / 50
    late = sum(history[-50:]) / 50
    assert late > early + 0.2, (
        f"sgd_ascent did not improve E[I] enough: early={early:.3f}, late={late:.3f}"
    )


# --------------------------------------------------------------------------- #
# sgd_descent on a sigmoid outage surrogate                                   #
# --------------------------------------------------------------------------- #


def test_sgd_descent_decreases_smooth_outage_probability() -> None:
    """Minimizing outage_probability_smooth via sgd_descent must lower it."""
    torch.manual_seed(0)
    d = 2
    P = 4.0
    R = 1.0
    tau = 0.1

    F = (0.1 * torch.randn(d, d, dtype=torch.complex128)).requires_grad_(True)
    H_sampler = samplers.rayleigh((d, d))
    sigma_x = torch.eye(d, dtype=torch.complex128)
    sigma_z = 0.5 * torch.eye(d, dtype=torch.complex128)

    def cost() -> torch.Tensor:
        K = compute_k_blocks_multiroot(
            num_nodes=2,
            roots=[0],
            parents={1: [0]},
            edge_mats={(1, 0): (H_sampler, F)},
            root_covs={0: sigma_x},
            noise_covs={1: sigma_z},
            batch_size=32,
        )
        I_samples = conditional_mutual_information_from_k(K, A=[0], B=[1]).real
        return outage_probability_smooth(I_samples, R=R, tau=tau)

    def projector(params: list[torch.Tensor]) -> list[torch.Tensor]:
        return [project_frobenius_ball(p, P=P) for p in params]

    history = sgd_descent(
        cost,
        [F],
        step_size=0.05,
        num_iters=300,
        projector=projector,
    )

    early = sum(history[:50]) / 50
    late = sum(history[-50:]) / 50
    # Smooth outage must decrease over training.
    assert late < early - 0.05, (
        f"sgd_descent did not reduce smooth outage enough: early={early:.3f}, "
        f"late={late:.3f}"
    )


# --------------------------------------------------------------------------- #
# Input validation                                                            #
# --------------------------------------------------------------------------- #


def test_sgd_ascent_zero_step_raises() -> None:
    F = torch.eye(2, dtype=torch.complex128).requires_grad_(True)
    with pytest.raises(ValueError, match="step_size"):
        sgd_ascent(lambda: F.real.sum(), [F], step_size=0.0, num_iters=10)


def test_sgd_ascent_zero_iters_raises() -> None:
    F = torch.eye(2, dtype=torch.complex128).requires_grad_(True)
    with pytest.raises(ValueError, match="num_iters"):
        sgd_ascent(lambda: F.real.sum(), [F], step_size=0.01, num_iters=0)


def test_sgd_ascent_non_grad_param_raises() -> None:
    F = torch.eye(2, dtype=torch.complex128)  # requires_grad=False
    with pytest.raises(ValueError, match="requires_grad"):
        sgd_ascent(lambda: F.real.sum(), [F], step_size=0.01, num_iters=10)


def test_sgd_descent_history_is_in_true_sign() -> None:
    """sgd_descent's returned history should be in the original (positive) sign."""
    F = torch.eye(2, dtype=torch.complex128).requires_grad_(True)

    def cost() -> torch.Tensor:
        # Trivial closure: depends on F via F.real.sum() squared.
        return (F.real.sum() - 5.0) ** 2

    history = sgd_descent(cost, [F], step_size=0.01, num_iters=20)
    # All entries should be non-negative (since cost is a squared error).
    assert all(h >= 0.0 for h in history)
