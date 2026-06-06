"""Stochastic (mini-batched) gradient ASCENT / DESCENT with optional projection.

The signature and history convention exactly mirror ``cmi_dag.pga_ascent`` and
``cmi_dag.pga_descent`` — the only behavioural difference is that the user's
closure is expected to **re-sample a fresh mini-batch** of channel realizations
on every call. With ``fading_dag.compute_k_blocks_multiroot`` doing the
re-sampling automatically (every edge sampler is called once per invocation),
the typical usage pattern is::

    batch_size = 64

    def compute_mi():
        K = compute_k_blocks_multiroot(..., batch_size=batch_size)
        I = conditional_mutual_information_from_k(K, A=[0], B=[2])
        return I.mean()             # E[I] surrogate for ergodic capacity

    history = sgd_ascent(
        compute_mi, [F], step_size=0.01, num_iters=200,
        projector=lambda ps: [project_frobenius_ball(p, P=5.0) for p in ps],
    )

For outage minimization, use ``sgd_descent`` with a smooth surrogate closure
(``outage_probability_smooth(I_samples, R, tau)``).

The closure and projector contracts are identical to the parent's
``pga_ascent`` / ``pga_descent``: a no-argument scalar-returning closure and
either an in-place mutating projector returning ``None`` or a functional
projector returning a sequence of new tensors.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch


def sgd_ascent(
    compute_mi: Callable[[], torch.Tensor],
    params: list[torch.Tensor],
    *,
    step_size: float,
    num_iters: int,
    projector: Callable[[list[torch.Tensor]], None | Sequence[torch.Tensor]] | None = None,
) -> list[float]:
    """Run projected stochastic gradient ASCENT on a scalar objective.

    At each iteration ``t = 0, 1, ..., num_iters - 1``:

        1. Zero out any existing gradients in ``params``.
        2. Compute ``loss = compute_mi()`` (which is expected to re-sample
           a fresh mini-batch of channel realizations internally), call
           ``loss.backward()``, and record ``loss.item()`` in the history.
        3. Inside ``torch.no_grad()``:
           a. Update each ``p in params`` via ``p.add_(step_size * p.grad)``.
           b. If ``projector`` is provided, invoke it. The projector may
              either mutate ``params`` in place (returning ``None``) or
              return a sequence of new tensors (one per parameter), which
              will be copied into place.

    Args:
        compute_mi: No-argument closure returning the scalar objective.
            Responsible for re-sampling the mini-batch on each call.
        params: List of leaf tensors with ``requires_grad=True``.
        step_size: Constant positive step size.
        num_iters: Number of SGD iterations (must be ``> 0``).
        projector: Optional functional or in-place projector onto the
            feasible set.

    Returns:
        ``history`` of length ``num_iters``; ``history[t]`` is the objective
        value recorded at the pre-update parameter state of iteration ``t``.
    """
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    if num_iters <= 0:
        raise ValueError(f"num_iters must be positive, got {num_iters}")
    for p in params:
        if not p.requires_grad:
            raise ValueError("All entries of `params` must have requires_grad=True.")

    history: list[float] = []
    for _ in range(num_iters):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        loss = compute_mi()
        loss.backward()
        history.append(loss.item())
        with torch.no_grad():
            for idx, p in enumerate(params):
                if p.grad is None:
                    raise RuntimeError(
                        f"params[{idx}] received no gradient after backward(): "
                        "the parameter has requires_grad=True but does not "
                        "participate in the autograd graph produced by "
                        "compute_mi(). Common causes: (a) the parameter is "
                        "declared but never used in the closure; (b) the "
                        "closure rebinds the parameter to a new tensor "
                        "(e.g. via `F = F.detach()` or in-place arithmetic "
                        "outside torch.no_grad); (c) a typo in a closure-"
                        "captured variable name. Verify that the parameter "
                        "is referenced inside compute_mi() and that the "
                        "returned objective tensor depends on it."
                    )
                p.add_(step_size * p.grad)
            if projector is not None:
                out = projector(params)
                if out is not None:
                    if len(out) != len(params):
                        raise ValueError(
                            f"projector returned {len(out)} tensors, expected "
                            f"{len(params)}."
                        )
                    for p, q in zip(params, out):
                        p.copy_(q)
    return history


def sgd_descent(
    closure: Callable[[], torch.Tensor],
    params: list[torch.Tensor],
    *,
    step_size: float,
    num_iters: int,
    projector: Callable[[list[torch.Tensor]], None | Sequence[torch.Tensor]] | None = None,
) -> list[float]:
    """Run projected stochastic gradient DESCENT on a cost-type objective.

    Minimization analog of ``sgd_ascent``: identical signature and history
    convention, but descends the objective. Internally negates the closure,
    forwards to ``sgd_ascent``, then flips the recorded history so values
    are reported in the true sign of the user's cost.

    Suitable for outage-probability minimization (with a sigmoid surrogate),
    MSE minimization, etc.
    """

    def negated_closure() -> torch.Tensor:
        return -closure()

    flipped_history = sgd_ascent(
        negated_closure,
        params,
        step_size=step_size,
        num_iters=num_iters,
        projector=projector,
    )
    return [-h for h in flipped_history]
