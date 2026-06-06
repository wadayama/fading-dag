"""Outage-minimizing MIMO precoder optimization under Rayleigh fading.

Problem:
    Channel: y = H F x + z   with H ~ CN(0, I) per realization (Rayleigh),
                                z ~ CN(0, sigma^2 I), x ~ CN(0, I).
    Objective: minimize the outage probability Pr[I(X; H F X + Z) < R]
               over the precoder F subject to ||F||_F^2 <= P.

Method:
    - During training, replace the (non-differentiable) raw outage with
      the smooth surrogate ``E[sigma((R - I)/tau)]``.
    - SGD-descent the surrogate under a Frobenius-ball projector that
      enforces the power constraint.
    - After training, evaluate the **raw** outage at a much larger batch
      to confirm the surrogate's gradient direction transfers to the
      true objective.

This problem is well-conditioned (precoder F directly scales the signal
without affecting the noise), so SGD reaches the budget boundary in only
a few hundred iterations and the outage drops by more than 10x.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from fading_dag import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
    outage_probability,
    outage_probability_smooth,
    project_frobenius_ball,
    samplers,
    sgd_descent,
)


def main() -> None:
    # Problem dimensions and rates.
    d = 2
    sigma2 = 1.0
    P = 5.0           # Frobenius budget on F.
    R = 1.5           # target rate (nats).
    tau = 0.3         # smoothing temperature.
    batch_size = 128
    num_iters = 500
    step_size = 0.1
    eval_batch = 10_000
    seed = 11

    torch.manual_seed(seed)
    F = (0.3 * torch.randn(d, d, dtype=torch.complex128)).requires_grad_(True)

    H_sampler = samplers.rayleigh((d, d))
    sigma_x = torch.eye(d, dtype=torch.complex128)
    sigma_z = sigma2 * torch.eye(d, dtype=torch.complex128)

    def smooth_outage_cost() -> torch.Tensor:
        K = compute_k_blocks_multiroot(
            num_nodes=2,
            roots=[0],
            parents={1: [0]},
            edge_mats={(1, 0): (H_sampler, F)},
            root_covs={0: sigma_x},
            noise_covs={1: sigma_z},
            batch_size=batch_size,
        )
        I = conditional_mutual_information_from_k(K, A=[0], B=[1]).real
        return outage_probability_smooth(I, R=R, tau=tau)

    def projector(params: list[torch.Tensor]) -> list[torch.Tensor]:
        return [project_frobenius_ball(p, P=P) for p in params]

    print(
        f"Starting sgd_descent: d={d}, sigma^2={sigma2}, P={P}, "
        f"R={R}, tau={tau}, B={batch_size}"
    )
    history = sgd_descent(
        smooth_outage_cost,
        [F],
        step_size=step_size,
        num_iters=num_iters,
        projector=projector,
    )

    print(f"Iter   0  smooth outage ~ {history[0]:.4f}")
    print(f"Iter {num_iters // 2}  smooth outage ~ {sum(history[num_iters//2 - 25 : num_iters//2 + 25]) / 50:.4f}")
    print(f"Iter {num_iters}  smooth outage ~ {sum(history[-50:]) / 50:.4f}")

    # Post-training evaluation: raw outage on a larger batch.
    with torch.no_grad():
        K_eval = compute_k_blocks_multiroot(
            num_nodes=2,
            roots=[0],
            parents={1: [0]},
            edge_mats={(1, 0): (H_sampler, F)},
            root_covs={0: sigma_x},
            noise_covs={1: sigma_z},
            batch_size=eval_batch,
        )
        I_eval = conditional_mutual_information_from_k(K_eval, A=[0], B=[1]).real
        raw_outage = outage_probability(I_eval, R=R).item()
        smooth_eval = outage_probability_smooth(I_eval, R=R, tau=tau).item()
        fnorm = torch.linalg.norm(F.detach()).item()

    print(
        f"Eval (B={eval_batch}): smooth outage = {smooth_eval:.4f}, "
        f"raw indicator = {raw_outage:.4f}"
    )
    print(f"||F_final||_F = {fnorm:.4f}  (budget sqrt(P) = {P**0.5:.4f})")

    # Plot trajectory.
    out_dir = Path(__file__).resolve().parent
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    iters = list(range(num_iters))
    ax.plot(iters, history, color="C3", alpha=0.4, label="smooth outage (raw)")
    window = 50
    if num_iters >= window:
        smoothed = [
            sum(history[max(0, t - window + 1) : t + 1])
            / min(t + 1, window)
            for t in range(num_iters)
        ]
        ax.plot(iters, smoothed, color="C3", linewidth=2.0, label=f"{window}-iter mean")
    ax.axhline(
        raw_outage, color="C0", linestyle="--", linewidth=1.0,
        label=f"raw outage @ B={eval_batch}: {raw_outage:.3f}",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Outage probability")
    ax.set_title(
        f"MIMO precoder outage minimization (Rayleigh, R={R}, tau={tau})"
    )
    ax.set_yscale("log")
    ax.grid(True, linewidth=0.4, which="both")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig_path = out_dir / "outage_minimizing_precoder.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved trajectory plot to {fig_path}")


if __name__ == "__main__":
    main()
