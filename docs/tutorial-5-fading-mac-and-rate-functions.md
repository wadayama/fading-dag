# Tutorial 5 — Fading MAC and rate functions

This capstone tutorial combines the multi-root machinery of `cmi-dag`
(Tutorials 1 and 3 there) with the fading-channel batching of
`fading-dag`. The model is a **2-user Rayleigh MAC**: two independent
transmitters with their own controllable precoders, one shared
receiver, both source-to-receiver links drawn fresh on every iteration.
The rate region's pentagon facets are computed *per realization*; we
contrast the **ergodic** rate region (using `E[f_T]`) with the
**outage-restricted** rate region (using `Pr[f_T < R_T]`) on the same
trained DAG.

By the end of this tutorial you will:

- Have built a 2-user fading MAC as a 3-node, 2-root DAG.
- Have evaluated the three pentagon facets *per channel realization*
  in one K-recursion forward pass.
- Have run `sgd_ascent` on the ergodic sum-rate of the pentagon and
  observed the rate region expand monotonically (in the trailing-mean
  sense).
- Understand how to compare ergodic and outage rate regions and what
  the two measure differently.
- Have a working pattern that scales to arbitrary multi-terminal
  channels (BC, IC, multi-hop relay, ...).

---

## 1. The model

The 2-user fading MAC:

```
   X_1  ──►  [ H_1 F_1 ]  ──┐
                            ├──►  Y = H_1 F_1 X_1 + H_2 F_2 X_2 + Z,
   X_2  ──►  [ H_2 F_2 ]  ──┘
                              X_k ~ CN(0, I_{d_k})  mutually independent,
                              Z   ~ CN(0, σ² I_{d_Y}),
                              H_k ~ CN(0, I_{d_Y} ⊗ I_{d_k})  drawn fresh per realization.
```

In DAG language this is a 3-node graph with **two roots**
(`V_0 = X_1`, `V_1 = X_2`) and one non-root sink (`V_2 = Y`). Each
source-to-receiver edge is a 2-tuple `(H_sampler_k, F_k)`:

- `H_sampler_k = samplers.rayleigh((d_Y, d_k))` — Rayleigh fading
  channel matrix.
- `F_k` — *trainable* precoder, shared across realizations.

The pentagon facets of the rate region are

```
I_1   = I(X_1; Y | X_2),
I_2   = I(X_2; Y | X_1),
I_12  = I(X_1, X_2; Y).
```

Per channel realization, each is a single
`conditional_mutual_information_from_k` call; aggregated, each becomes
either an *ergodic* or an *outage* quantity.

---

## 2. Build the DAG and compute the pentagon facets per realization

```python
import torch
from fading_dag import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
    samplers,
)

torch.manual_seed(0)
d, sigma2, B = 2, 0.5, 200
DTYPE = torch.complex128

# Initialise (will be trained in §3).
F_1 = (0.5 * torch.randn(d, d, dtype=DTYPE)).requires_grad_(True)
F_2 = (0.5 * torch.randn(d, d, dtype=DTYPE)).requires_grad_(True)

H_1_sampler = samplers.rayleigh((d, d))
H_2_sampler = samplers.rayleigh((d, d))

edge_mats = {
    (2, 0): (H_1_sampler, F_1),
    (2, 1): (H_2_sampler, F_2),
}
root_covs  = {0: torch.eye(d, dtype=DTYPE), 1: torch.eye(d, dtype=DTYPE)}
noise_covs = {2: sigma2 * torch.eye(d, dtype=DTYPE)}

K = compute_k_blocks_multiroot(
    num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
    edge_mats=edge_mats,
    root_covs=root_covs,
    noise_covs=noise_covs,
    batch_size=B,
)

I_1   = conditional_mutual_information_from_k(K, A=[0],    B=[2], C=[1])
I_2   = conditional_mutual_information_from_k(K, A=[1],    B=[2], C=[0])
I_12  = conditional_mutual_information_from_k(K, A=[0, 1], B=[2], C=[])

print(f"per-realization shapes: {tuple(I_1.shape)}, {tuple(I_2.shape)}, {tuple(I_12.shape)}")
print(f"ergodic estimates:")
print(f"  E[I_1 ] ≈ {I_1.real.mean().item():.4f} nats")
print(f"  E[I_2 ] ≈ {I_2.real.mean().item():.4f} nats")
print(f"  E[I_12] ≈ {I_12.real.mean().item():.4f} nats")
```

The three values are PyTorch tensors of shape `(B,)` — one entry per
channel realization. The K-recursion forward pass is shared across all
three CMI calls (you only build `K` once).

> **Sanity:** by the chain rule of MI,
> `I_1 + I(X_2; Y) = I_12` *per realization*. Try computing
> `I(X_2; Y)` with `C = ()` and check the identity at every batch index.
> The cmi-dag test suite checks this in
> [`tests/test_conditional_information.py::test_pentagon_chain_rule`](https://github.com/wadayama/cmi-dag/blob/main/tests/test_conditional_information.py).

---

## 3. Express the pentagon as a list of rate functions

A rate function is a linear combination of CMIs. Each one becomes a
list of `Summand` tuples `(alpha, A, B, C)`. For the MAC pentagon all
three coefficients are `+1`:

```python
from fading_dag import evaluate_rate_functions

pentagon = [
    [(1.0, [0],    [2], [1])],     # f_1   = I_1
    [(1.0, [1],    [2], [0])],     # f_2   = I_2
    [(1.0, [0, 1], [2], [])],      # f_12  = I_12
]

f_per_realization = evaluate_rate_functions(K, pentagon)
print(f"f_per_realization shapes: " + ", ".join(str(tuple(f.shape)) for f in f_per_realization))
```

`evaluate_rate_functions` produces a list of `(B,)` tensors — one per
rate function, all evaluated from the *same* K-blocks (one K-recursion
forward pass shared across the entire family). This generalizes to any
number of facets, sign-mixed coefficients (HK / wiretap), and rate
functions with `N > 1` summands.

---

## 4. Maximize the ergodic sum-rate

The simplest aggregation is to add the three ergodic facet values into
a single scalar surrogate

```
J_ergodic(F_1, F_2) = E[f_1] + E[f_2] + E[f_12],
```

and ascend it under a *shared* total-power budget. The
`project_total_power` projector enforces `||F_1||_F^2 + ||F_2||_F^2 ≤ P`:

```python
from fading_dag import project_total_power, sgd_ascent

P_total = 8.0

def ergodic_pentagon_sum() -> torch.Tensor:
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats=edge_mats,
        root_covs=root_covs,
        noise_covs=noise_covs,
        batch_size=B,
    )
    f_1, f_2, f_12 = evaluate_rate_functions(K, pentagon)
    return f_1.real.mean() + f_2.real.mean() + f_12.real.mean()

history = sgd_ascent(
    ergodic_pentagon_sum,
    [F_1, F_2],
    step_size=0.02,
    num_iters=300,
    projector=lambda ps: project_total_power(ps, P_total),
)
print(f"ergodic pentagon sum: {history[0]:.3f} -> {history[-1]:.3f} nats")
print(f"||F_1||_F^2 + ||F_2||_F^2 = "
      f"{torch.linalg.norm(F_1.detach()).item()**2 + torch.linalg.norm(F_2.detach()).item()**2:.3f} (budget {P_total})")
```

The sum rises (in trailing-window mean), and the two precoder
magnitudes saturate the shared budget. Because the two channels are
i.i.d. Rayleigh and the model is symmetric, optimal symmetric
allocation gives `||F_1||_F^2 ≈ ||F_2||_F^2 ≈ P_total / 2`. (Asymmetric
solutions are possible with asymmetric channels; the library does not
impose symmetry.)

---

## 5. Evaluate the trained ergodic rate region

The training surrogate is a scalar; the *rate region* is the
intersection of three facet inequalities. Evaluate each facet at the
trained `F_1, F_2` on a large evaluation batch:

```python
with torch.no_grad():
    K_eval = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats=edge_mats,                                  # F_1, F_2 are still trained leaves
        root_covs=root_covs,
        noise_covs=noise_covs,
        batch_size=10_000,
    )
    f_1, f_2, f_12 = evaluate_rate_functions(K_eval, pentagon)
    E_f_1   = f_1.real.mean().item()
    E_f_2   = f_2.real.mean().item()
    E_f_12  = f_12.real.mean().item()

print(f"ergodic rate region (estimated at B=10k):")
print(f"  R_1   ≤ {E_f_1:.4f} nats")
print(f"  R_2   ≤ {E_f_2:.4f} nats")
print(f"  R_1+R_2 ≤ {E_f_12:.4f} nats")
print(f"  pentagon corner (sum-rate-maximizing): "
      f"R_1* = {E_f_12 - E_f_2:.4f}, R_2* = {E_f_12 - E_f_1:.4f}")
```

These three numbers carve out the **ergodic** pentagon: the set of
average-rate pairs achievable on the fading MAC under the chosen
precoders. The pentagon corner is the operating point that maximizes
the sum rate while saturating both single-user facets.

---

## 6. Outage rate region

If your application is delay-sensitive (real-time video, control
signaling), the ergodic pentagon is the wrong metric. You want each
facet to hold *with high probability per realization*: there is some
target rate vector `(R_1, R_2)` and you want

```
Pr[ f_1 < R_1 ] ≤ ε_1,    Pr[ f_2 < R_2 ] ≤ ε_2,    Pr[ f_12 < R_1 + R_2 ] ≤ ε_12.
```

The library's outage helpers operate facet by facet:

```python
from fading_dag import outage_probability

R_target = (E_f_1 * 0.6, E_f_2 * 0.6)               # 60% of trained ergodic facets
out_1  = outage_probability(f_1.real,  R=R_target[0]).item()
out_2  = outage_probability(f_2.real,  R=R_target[1]).item()
out_12 = outage_probability(f_12.real, R=R_target[0] + R_target[1]).item()
print(f"outage at 60% of ergodic facets:")
print(f"  Pr[f_1 < R_1]      ≈ {out_1:.4f}")
print(f"  Pr[f_2 < R_2]      ≈ {out_2:.4f}")
print(f"  Pr[f_12 < R_1+R_2] ≈ {out_12:.4f}")
```

For a *trained-for-ergodic* DAG, the outage numbers will typically be
non-trivial (10–30%) — the precoders were not designed with tail
behavior in mind. To improve them, switch the training surrogate to a
weighted sum of `outage_probability_smooth` calls and descend:

```python
from fading_dag import outage_probability_smooth, sgd_descent

R_1, R_2, R_12 = 0.6 * E_f_1, 0.6 * E_f_2, 0.6 * (E_f_1 + E_f_2)
tau = 0.3
weights = (1.0, 1.0, 1.0)

def outage_pentagon_cost() -> torch.Tensor:
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats=edge_mats,
        root_covs=root_covs,
        noise_covs=noise_covs,
        batch_size=B,
    )
    f_1, f_2, f_12 = evaluate_rate_functions(K, pentagon)
    out_1  = outage_probability_smooth(f_1.real,  R=R_1, tau=tau)
    out_2  = outage_probability_smooth(f_2.real,  R=R_2, tau=tau)
    out_12 = outage_probability_smooth(f_12.real, R=R_12, tau=tau)
    return weights[0] * out_1 + weights[1] * out_2 + weights[2] * out_12

history_out = sgd_descent(
    outage_pentagon_cost,
    [F_1, F_2],
    step_size=0.05,
    num_iters=500,
    projector=lambda ps: project_total_power(ps, P_total),
)
print(f"outage cost: {history_out[0]:.4f} -> {history_out[-1]:.4f}")
```

After training, the *trained-for-outage* DAG should give smaller
`Pr[f_T < R_T]` values than the ergodic-trained one at the same
`(R_1, R_2)` target — at the cost of a smaller `E[f_T]` (the
ergodic/outage tradeoff).

---

## 7. The bigger picture

The pattern this tutorial walks through generalizes directly:

- **Any multi-terminal channel.** Add roots, add parents, add edges.
  The K-recursion handles arbitrary multi-root DAGs; the rate-function
  evaluator handles arbitrary linear combinations of CMIs with arbitrary
  signs.
- **Any aggregator.** `E[f_T]`, `Pr[f_T < R_T]` and its smooth
  surrogate are the three aggregators in `fading_dag.outage`. Adding a
  variance term `Var(f_T)` or a quantile estimator is a few-line
  custom helper.
- **Any sampler.** Block fading, time-correlated fading, measured
  channel traces — all conform to `Callable[[int], Tensor]` and slot
  into `edge_mats` without changing the rest of the pipeline (Tutorial 2).
- **Any projector.** Frobenius ball per precoder, shared total-power
  budget, Stiefel manifold (unitary), low-rank truncation via SVD —
  all of these compose with `sgd_ascent` and `sgd_descent` via the
  `projector` argument.

The remaining ICC 2027 paper-style applications — fading-aware
amplify-and-forward relays, RIS phase-shift design under correlated
fading, secrecy-rate optimization on a wiretap channel under
block-fading — all follow the same pattern. None of them require new
library primitives.

---

## 8. Where to go from here

- Read the **Public API** table in the [README](../README.md) for a
  reference catalog of every symbol exposed at the top of
  `fading_dag`.
- Skim the **Conventions** section in the [README](../README.md) for
  edge-cases (Hermitian flip on K-blocks, complex autograd's
  factor-of-2, jitter for near-singular conditional covariances).
- Browse the **theoretical-validation tests** in
  [`tests/test_theoretical_validation.py`](../tests/test_theoretical_validation.py)
  for runnable examples of closed-form sanity checks (Telatar's
  ergodic capacity, SISO outage, constant-channel limit, SGD reaching
  the closed-form Rayleigh-SISO optimum).
- For the deterministic-channel building blocks (single-root
  K-recursion, single-pair MI, projected gradient ascent), read the
  parent libraries
  [`gaussian-dag`](https://github.com/wadayama/gaussian-dag) and
  [`cmi-dag`](https://github.com/wadayama/cmi-dag). Their tutorial
  series introduces the K-recursion and PGA from scratch and is the
  natural prerequisite for the fading-channel material here.

You should now be able to set up, train, and evaluate any
linear-Gaussian-DAG fading-channel problem the library is in scope for.
Welcome to `fading-dag`!
