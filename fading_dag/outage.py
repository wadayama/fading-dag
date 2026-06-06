"""Outage probability and ergodic capacity helpers.

These thin aggregators consume the per-realization mutual-information samples
returned by
``fading_dag.conditional_mutual_information_from_k`` /
``fading_dag.evaluate_rate_functions`` and produce the standard fading-channel
performance measures.

For optimization, outage probability is approximated by a *sigmoid surrogate*:

    outage_smooth(I) = E[sigma((R - I) / tau)]

which converges to the raw indicator probability ``E[1[I < R]]`` as
``tau -> 0`` while remaining differentiable for any ``tau > 0``. The raw
indicator version (``outage_probability``) is provided alongside for
evaluation / reporting once optimization has converged.

Convenience::

    ergodic_capacity(I_samples) == I_samples.mean()
"""

from __future__ import annotations

import torch


def ergodic_capacity(I_samples: torch.Tensor) -> torch.Tensor:
    """Empirical ergodic capacity (mini-batch mean of MI samples).

    Differentiable through ``I_samples`` so it can be used directly as a
    SGD loss for ergodic-capacity maximization.

    Args:
        I_samples: Per-realization MI tensor of shape ``(B,)`` (or any
            leading shape; mean is taken over all entries).

    Returns:
        Scalar tensor: mean of ``I_samples``.
    """
    if I_samples.numel() == 0:
        raise ValueError("I_samples is empty; cannot compute ergodic capacity.")
    return I_samples.mean()


def outage_probability(I_samples: torch.Tensor, R: float) -> torch.Tensor:
    """Empirical outage probability ``Pr[I < R]`` via the raw indicator.

    Computes ``(I_samples < R).float().mean()``. Suitable for **evaluation**
    after optimization (or for reporting): the indicator function is *not*
    differentiable, so this scalar is unfit for backprop.

    Args:
        I_samples: Per-realization MI tensor of shape ``(B,)`` (real-valued).
        R: Threshold rate (typically in nats; consistent with the unit of
            ``I_samples``).

    Returns:
        Scalar real tensor in ``[0, 1]``: empirical Pr[I < R]. Detached
        from the autograd graph because the indicator is non-differentiable.
    """
    if I_samples.numel() == 0:
        raise ValueError("I_samples is empty; cannot compute outage probability.")
    indicator = (I_samples.detach() < R).to(I_samples.real.dtype)
    return indicator.mean()


def outage_probability_smooth(
    I_samples: torch.Tensor,
    R: float,
    tau: float,
) -> torch.Tensor:
    """Differentiable sigmoid surrogate of the outage probability.

    Returns ``E[sigma((R - I) / tau)]`` where ``sigma`` is the logistic
    sigmoid. As ``tau -> 0`` the surrogate converges to the raw indicator
    probability ``E[1[I < R]]``; finite ``tau > 0`` provides a smooth,
    differentiable scalar suitable for SGD-based outage minimization.

    .. note::
        The gradient flows through ``sigma'((R - I)/tau)`` and **vanishes
        when every realization sits far on one side of the threshold**.
        Two common pitfalls:
        (a) initializing the controllable factor at a tiny magnitude, so
        that ``I`` is essentially zero everywhere and ``sigma' ≈ 0``;
        (b) choosing ``tau`` too small from the start, so that any
        realization away from ``R`` contributes zero gradient.
        Remedies: initialize at a moderate magnitude (e.g. half the
        Frobenius budget), start with ``tau`` in ``[0.3, 0.5]``, and use a
        large enough mini-batch. See the README's
        "sigmoid saturation" section for the full discussion.

    Args:
        I_samples: Per-realization MI tensor of shape ``(B,)``.
        R: Threshold rate.
        tau: Smoothing temperature (strictly positive). Smaller ``tau``
            yields a sharper approximation but larger gradients (and a
            higher risk of vanishing-gradient saturation; see the note
            above). Practical values lie in ``[0.05, 0.5]`` for ``I`` in
            nats.

    Returns:
        Scalar real tensor in ``(0, 1)``, differentiable in ``I_samples``.

    Raises:
        ValueError: if ``tau <= 0`` or ``I_samples`` is empty.
    """
    if tau <= 0:
        raise ValueError(f"tau must be strictly positive; got {tau}")
    if I_samples.numel() == 0:
        raise ValueError(
            "I_samples is empty; cannot compute smooth outage probability."
        )
    return torch.sigmoid((R - I_samples) / tau).mean()
