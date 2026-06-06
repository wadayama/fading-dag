"""Validation tests against closed-form theoretical values.

For SISO Rayleigh fading (single-antenna point-to-point with unit-variance
i.i.d. CN(0, 1) channel coefficient h, unit-variance noise) the relevant
information-theoretic quantities admit closed forms in nats:

    Channel:    y = h x + z,  h ~ CN(0, 1),  z ~ CN(0, 1),  E|x|^2 = gamma.
    Per-realization MI: I(h) = log(1 + gamma * |h|^2).

Quantities verified here:

1. **Ergodic capacity** (Telatar 1999, eq. 4 for the scalar case):

        C_erg = E_h[ I(h) ] = exp(1/gamma) * E_1(1/gamma)   (nats)

   where E_1(x) = int_x^infty (e^{-t} / t) dt is the exponential integral.

2. **Outage probability** at threshold R (pure closed form, no special
   functions needed):

        Pr[I(h) < R]
            = Pr[|h|^2 < (e^R - 1) / gamma]
            = 1 - exp(-(e^R - 1) / gamma)        (since |h|^2 ~ Exp(1)).

3. **Constant channel (no fading)**: with the ``constant`` sampler in place
   of a Rayleigh sampler, every per-realization MI must match the
   deterministic log-det MI

        I_det = log det(I + sigma^{-2} H F Sigma_X F^H H^H)

   to machine precision.

These tests rely on a finite Monte Carlo sample, so the empirical estimates
are compared against the closed forms with a tolerance large enough to
cover the standard Monte Carlo error at the chosen batch size (3-4 sigma),
while still being tight enough to catch genuine systematic errors.
"""

from __future__ import annotations

import math

import pytest
import torch
from scipy.special import exp1  # type: ignore[import-not-found]

from fading_dag import samplers
from fading_dag.information import (
    conditional_mutual_information_from_k,
    logdet_hpd,
)
from fading_dag.krecursion import compute_k_blocks_multiroot
from fading_dag.outage import outage_probability


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _siso_rayleigh_mi_samples(
    gamma: float,
    *,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    """Per-realization MI samples for the unit-variance SISO Rayleigh model.

    Channel:  y = h x + z   with  h ~ CN(0, 1),  z ~ CN(0, 1),  E|x|^2 = gamma.

    Returns:
        Real tensor of shape (batch_size,), per-realization log(1+gamma*|h|^2).
    """
    torch.manual_seed(seed)
    Sigma_X = torch.tensor([[complex(gamma, 0.0)]], dtype=torch.complex128)
    Sigma_Z = torch.tensor([[complex(1.0, 0.0)]], dtype=torch.complex128)
    F = torch.eye(1, dtype=torch.complex128)
    H_sampler = samplers.rayleigh((1, 1))

    K = compute_k_blocks_multiroot(
        num_nodes=2,
        roots=[0],
        parents={1: [0]},
        edge_mats={(1, 0): (H_sampler, F)},
        root_covs={0: Sigma_X},
        noise_covs={1: Sigma_Z},
        batch_size=batch_size,
    )
    return conditional_mutual_information_from_k(K, A=[0], B=[1]).real


def _siso_rayleigh_ergodic_capacity_nats(gamma: float) -> float:
    """Closed-form SISO Rayleigh ergodic capacity in nats."""
    inv = 1.0 / gamma
    return math.exp(inv) * float(exp1(inv))


def _siso_rayleigh_outage_probability(gamma: float, R: float) -> float:
    """Closed-form SISO Rayleigh outage probability Pr[log(1+gamma|h|^2) < R]."""
    return 1.0 - math.exp(-(math.exp(R) - 1.0) / gamma)


# --------------------------------------------------------------------------- #
# 1. SISO Rayleigh ergodic capacity                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("gamma", [1.0, 5.0, 10.0])
def test_siso_rayleigh_ergodic_capacity_matches_telatar(gamma: float) -> None:
    """Empirical E[I] over a large mini-batch must match exp(1/gamma) E_1(1/gamma).

    Tolerance bands cover ~4-sigma Monte Carlo error at B = 50_000.
    """
    B = 50_000
    I = _siso_rayleigh_mi_samples(gamma=gamma, batch_size=B, seed=0)
    empirical = I.mean().item()
    theoretical = _siso_rayleigh_ergodic_capacity_nats(gamma)

    # Per-sample MI std is bounded; empirical std at B=50k gives an
    # estimated mean stderr of sd/sqrt(B). 0.05 absolute tolerance is
    # several stderr wide for all practical gamma in [1, 10].
    assert abs(empirical - theoretical) < 0.05, (
        f"SISO Rayleigh ergodic capacity mismatch at gamma={gamma}: "
        f"empirical={empirical:.4f}, theoretical={theoretical:.4f}, "
        f"gap={abs(empirical - theoretical):.4f}"
    )


# --------------------------------------------------------------------------- #
# 2. SISO Rayleigh outage probability                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("gamma", "R"),
    [
        (5.0, 0.5),
        (5.0, 1.0),
        (5.0, 1.5),
        (10.0, 1.0),
        (1.0, 0.3),
    ],
)
def test_siso_rayleigh_outage_matches_closed_form(gamma: float, R: float) -> None:
    """Empirical Pr[I < R] must match 1 - exp(-(e^R - 1)/gamma).

    Binomial standard error at B = 100_000 is at most 0.5/sqrt(B) ~ 0.0016;
    a 0.01 absolute tolerance comfortably covers ~6-sigma Monte Carlo error.
    """
    B = 100_000
    I = _siso_rayleigh_mi_samples(gamma=gamma, batch_size=B, seed=1)
    empirical = outage_probability(I, R=R).item()
    theoretical = _siso_rayleigh_outage_probability(gamma=gamma, R=R)

    assert abs(empirical - theoretical) < 0.01, (
        f"SISO Rayleigh outage mismatch at gamma={gamma}, R={R}: "
        f"empirical={empirical:.4f}, theoretical={theoretical:.4f}, "
        f"gap={abs(empirical - theoretical):.4f}"
    )


# --------------------------------------------------------------------------- #
# 3. Constant channel (no fading): per-realization MI = deterministic log-det  #
# --------------------------------------------------------------------------- #


def test_constant_channel_matches_deterministic_log_det() -> None:
    """With `samplers.constant`, every per-realization MI must equal the
    deterministic log-det MI to machine precision.

    Deterministic MI for y = H F x + z with x ~ CN(0, Sigma_X), z ~ CN(0,
    Sigma_Z):

        I = log det(Sigma_Y) - log det(Sigma_{Y|X})
          = log det(Sigma_Z + H F Sigma_X F^H H^H) - log det(Sigma_Z).

    For Sigma_X = I and Sigma_Z = sigma^2 I this reduces to the textbook
    capacity log det(I + sigma^{-2} H F F^H H^H).
    """
    torch.manual_seed(7)
    d = 3
    sigma2 = 0.25
    B = 8

    H_fixed = torch.randn(d, d, dtype=torch.complex128)
    F = torch.randn(d, d, dtype=torch.complex128)
    Sigma_X = torch.eye(d, dtype=torch.complex128)
    Sigma_Z = sigma2 * torch.eye(d, dtype=torch.complex128)

    K = compute_k_blocks_multiroot(
        num_nodes=2,
        roots=[0],
        parents={1: [0]},
        edge_mats={(1, 0): (samplers.constant(H_fixed), F)},
        root_covs={0: Sigma_X},
        noise_covs={1: Sigma_Z},
        batch_size=B,
    )
    I_samples = conditional_mutual_information_from_k(K, A=[0], B=[1]).real

    # Deterministic ground truth.
    A = H_fixed @ F
    Sigma_Y = Sigma_Z + A @ Sigma_X @ A.mH
    I_det = (logdet_hpd(Sigma_Y) - logdet_hpd(Sigma_Z)).item()

    # Every batch entry must coincide with I_det.
    diffs = (I_samples - I_det).abs()
    max_diff = diffs.max().item()
    assert max_diff < 1e-10, (
        f"Constant-channel MI does not match deterministic log-det: "
        f"max |I_sample - I_det| = {max_diff:.3e}, I_det = {I_det:.6f}"
    )
    # And all samples are identical (since H is constant across batch).
    intra_batch_spread = (I_samples - I_samples[0]).abs().max().item()
    assert intra_batch_spread < 1e-12, (
        f"Constant-channel MI samples disagree across batch: "
        f"spread = {intra_batch_spread:.3e}"
    )


# --------------------------------------------------------------------------- #
# 4. Outage from the smooth surrogate converges to closed-form in tau -> 0    #
# --------------------------------------------------------------------------- #


def test_outage_smooth_converges_to_closed_form_as_tau_decreases() -> None:
    """outage_probability_smooth(I_samples, R, tau) -> closed-form as tau -> 0."""
    from fading_dag.outage import outage_probability_smooth

    gamma = 5.0
    R = 1.0
    B = 100_000
    I = _siso_rayleigh_mi_samples(gamma=gamma, batch_size=B, seed=2)
    theoretical = _siso_rayleigh_outage_probability(gamma=gamma, R=R)

    smooth_coarse = outage_probability_smooth(I, R=R, tau=0.5).item()
    smooth_fine = outage_probability_smooth(I, R=R, tau=0.02).item()
    # As tau decreases, the surrogate must get closer to the closed-form.
    assert abs(smooth_fine - theoretical) < abs(smooth_coarse - theoretical), (
        f"smooth outage did not converge toward closed form as tau decreased: "
        f"|smooth(tau=0.5) - theo| = {abs(smooth_coarse - theoretical):.4f}, "
        f"|smooth(tau=0.02) - theo| = {abs(smooth_fine - theoretical):.4f}"
    )
    # And the fine surrogate should match the closed form within MC tolerance.
    assert abs(smooth_fine - theoretical) < 0.015, (
        f"smooth outage (tau=0.02) does not match closed form: "
        f"smooth={smooth_fine:.4f}, theoretical={theoretical:.4f}"
    )


# --------------------------------------------------------------------------- #
# 5. SGD on smooth-outage surrogate reaches the closed-form SISO Rayleigh     #
#    optimum.                                                                 #
# --------------------------------------------------------------------------- #


def test_sgd_descent_reaches_closed_form_siso_outage_minimum() -> None:
    """SGD on the smooth-outage surrogate must reach the Rayleigh SISO optimum.

    Setup (SISO, unit-variance noise):
        Channel: y = h * F * x + z,  h ~ CN(0, 1), z ~ CN(0, 1), x ~ CN(0, 1).
        Budget: |F|^2 <= P.

    The Rayleigh SISO MI per realization is

        I(h) = log(1 + |h|^2 * |F|^2),

    which is monotone in |F|^2. The optimal precoder therefore uses full
    power, |F|^2 = P, and the resulting outage is the closed form

        Pr[I < R]_min = 1 - exp(-(e^R - 1) / P).

    The test asserts:
    (a) SGD drives ||F||_F to the Frobenius-ball boundary (full power),
    (b) the trained model's raw outage matches the closed-form optimum
        within ~2 percentage points (MC tolerance at the chosen eval batch).
    """
    import math

    from fading_dag import (
        compute_k_blocks_multiroot,
        conditional_mutual_information_from_k,
        outage_probability,
        outage_probability_smooth,
        project_frobenius_ball,
        samplers,
        sgd_descent,
    )

    d = 1
    P = 5.0
    R = 1.0
    tau = 0.3       # broader sigmoid -> avoid early saturation near small init.
    batch_size = 256
    num_iters = 500
    step_size = 0.1
    eval_batch = 20_000
    seed = 3

    torch.manual_seed(seed)
    # Initialize at a moderate magnitude (1/2 the budget). With very small
    # init the sigmoid surrogate saturates and gradients vanish; with a
    # near-boundary init the optimizer only needs to refine the direction.
    F_init = math.sqrt(P) / 2 * torch.randn(d, d, dtype=torch.complex128)
    F_init = F_init / torch.linalg.norm(F_init) * (math.sqrt(P) / 2)
    F = F_init.clone().requires_grad_(True)
    Sigma_X = torch.eye(d, dtype=torch.complex128)
    Sigma_Z = torch.eye(d, dtype=torch.complex128)
    H_sampler = samplers.rayleigh((d, d))

    def closure() -> torch.Tensor:
        K = compute_k_blocks_multiroot(
            num_nodes=2,
            roots=[0],
            parents={1: [0]},
            edge_mats={(1, 0): (H_sampler, F)},
            root_covs={0: Sigma_X},
            noise_covs={1: Sigma_Z},
            batch_size=batch_size,
        )
        I = conditional_mutual_information_from_k(K, A=[0], B=[1]).real
        return outage_probability_smooth(I, R=R, tau=tau)

    def projector(params: list[torch.Tensor]) -> list[torch.Tensor]:
        return [project_frobenius_ball(p, P=P) for p in params]

    sgd_descent(closure, [F], step_size=step_size, num_iters=num_iters,
                 projector=projector)

    # (a) F should be at the Frobenius-ball boundary (||F||_F = sqrt(P)).
    fnorm = torch.linalg.norm(F.detach()).item()
    assert abs(fnorm - math.sqrt(P)) < 1e-3, (
        f"||F||_F did not reach the Frobenius-ball boundary: "
        f"||F||={fnorm:.4f}, sqrt(P)={math.sqrt(P):.4f}"
    )

    # (b) Raw outage at convergence must match the closed-form optimum.
    K_eval = compute_k_blocks_multiroot(
        num_nodes=2,
        roots=[0],
        parents={1: [0]},
        edge_mats={(1, 0): (H_sampler, F.detach())},
        root_covs={0: Sigma_X},
        noise_covs={1: Sigma_Z},
        batch_size=eval_batch,
    )
    I_eval = conditional_mutual_information_from_k(K_eval, A=[0], B=[1]).real
    raw_final = outage_probability(I_eval, R=R).item()

    theoretical = _siso_rayleigh_outage_probability(gamma=P, R=R)
    assert abs(raw_final - theoretical) < 0.02, (
        f"Trained SISO Rayleigh raw outage does not match closed-form optimum: "
        f"trained raw outage = {raw_final:.4f}, "
        f"theoretical = {theoretical:.4f}, "
        f"gap = {abs(raw_final - theoretical):.4f}"
    )
