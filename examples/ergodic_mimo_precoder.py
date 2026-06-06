"""Ergodic MIMO precoder optimization under Rayleigh fading.

Problem:
    Channel: y = H F x + z   with H ~ CN(0, I) per realization (Rayleigh),
                                z ~ CN(0, sigma^2 I), x ~ CN(0, I).
    Objective: maximize ergodic capacity E_H[I(X; H F X + Z)]
               over the precoder F subject to ||F||_F^2 <= P.

Method:
    - Each iteration draws a fresh batch of `batch_size` channel realizations
      via the Rayleigh sampler; the batched K-recursion then yields a
      per-realization MI tensor of shape (B,).
    - The mean is the SGD surrogate for ergodic capacity, and projected SGD
      with `project_frobenius_ball` enforces the power budget.

Plot:
    Trajectory of the trailing-window mean MI vs iteration, saved next to
    this script.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from fading_dag import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
    project_frobenius_ball,
    samplers,
    sgd_ascent,
)


def main() -> None:
    # Problem dimensions.
    d = 2
    sigma2 = 0.5
    P = 4.0  # Frobenius power budget on F.
    batch_size = 64
    num_iters = 600
    step_size = 0.05
    seed = 7

    torch.manual_seed(seed)

    # Controllable precoder F (the design variable).
    F = (0.2 * torch.randn(d, d, dtype=torch.complex128)).requires_grad_(True)

    # Edge specification: (H_sampler, F).
    H_sampler = samplers.rayleigh((d, d))
    edge_mats = {(1, 0): (H_sampler, F)}

    sigma_x = torch.eye(d, dtype=torch.complex128)
    sigma_z = sigma2 * torch.eye(d, dtype=torch.complex128)

    def compute_mi() -> torch.Tensor:
        K = compute_k_blocks_multiroot(
            num_nodes=2,
            roots=[0],
            parents={1: [0]},
            edge_mats=edge_mats,
            root_covs={0: sigma_x},
            noise_covs={1: sigma_z},
            batch_size=batch_size,
        )
        I = conditional_mutual_information_from_k(K, A=[0], B=[1]).real
        return I.mean()  # ergodic capacity surrogate

    def projector(params: list[torch.Tensor]) -> list[torch.Tensor]:
        return [project_frobenius_ball(p, P=P) for p in params]

    print(f"Starting sgd_ascent: d={d}, sigma2={sigma2}, P={P}, B={batch_size}")
    history = sgd_ascent(
        compute_mi,
        [F],
        step_size=step_size,
        num_iters=num_iters,
        projector=projector,
    )

    print(f"Iter   0  E[I] ~ {history[0]:.4f} nats")
    print(f"Iter {num_iters // 2}  E[I] ~ {sum(history[num_iters//2 - 25 : num_iters//2 + 25]) / 50:.4f} nats")
    print(f"Iter {num_iters}  E[I] ~ {sum(history[-50:]) / 50:.4f} nats")

    # Final precoder summary.
    with torch.no_grad():
        F_final = F.detach()
        print(f"||F_final||_F = {torch.linalg.norm(F_final).item():.4f}  (budget {math.sqrt(P):.4f})")

    # Plot trajectory.
    out_dir = Path(__file__).resolve().parent
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    iters = list(range(num_iters))
    ax.plot(iters, history, color="C0", alpha=0.4, label="raw MI (B-mini-batch)")
    # 50-iter rolling mean for visual smoothing.
    window = 50
    if num_iters >= window:
        smoothed = [
            sum(history[max(0, t - window + 1) : t + 1])
            / min(t + 1, window)
            for t in range(num_iters)
        ]
        ax.plot(iters, smoothed, color="C0", linewidth=2.0, label=f"{window}-iter mean")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("E[I] (nats)")
    ax.set_title(f"Ergodic MIMO precoder under Rayleigh fading (d={d}, P={P})")
    ax.grid(True, linewidth=0.4)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig_path = out_dir / "ergodic_mimo_precoder.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved trajectory plot to {fig_path}")


if __name__ == "__main__":
    main()
