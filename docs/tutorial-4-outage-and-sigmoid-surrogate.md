# Tutorial 4 — Outage probability and the sigmoid surrogate

Ergodic capacity (Tutorial 3) is the *expected* mutual information; it
tells you how a link will perform *on average*. Outage probability is
the dual quantity: the probability that the realized MI drops below a
target rate `R`, capturing the *tail* behavior. Both quantities are
fundamental, but optimizing outage probability is structurally harder
because the natural estimator is non-differentiable. This tutorial
introduces the **sigmoid surrogate** that makes the problem
back-propagatable, the `sgd_descent` companion to `sgd_ascent`, and the
**sigmoid saturation** pitfall that you will inevitably hit if you try
to optimize outage without the surrogate's three standard remedies.

By the end of this tutorial you will:

- Understand why `outage_probability(I_samples, R)` cannot be used as
  a training loss as-is.
- Understand the sigmoid surrogate `E[sigma((R - I) / tau)]` and the
  role of the temperature `tau`.
- Be able to run a full outage-minimization SGD loop using
  `sgd_descent` and `outage_probability_smooth`.
- Recognize the three saturation pitfalls (small `F`, small `tau`,
  high-noise channel) and apply the standard remedies.
- Use the `(smooth surrogate during training, raw indicator during
  evaluation)` split that the test suite already enforces.

---

## 1. The raw indicator is correct but unusable

The outage probability is

```
P_out(F, R) = Pr_H[ I(X; H F X + Z) < R ].
```

Estimated on a `B`-sample mini-batch, it is the empirical CDF

```
P̂_out(F, R) = (1/B) · Σ_{b=1}^{B} 1[ I^{(b)}(F) < R ].
```

`fading_dag.outage.outage_probability(I_samples, R)` computes exactly
this. It is a one-line wrapper around `(I_samples < R).float().mean()`,
and it *detaches* its input from the autograd graph so that you cannot
accidentally back-propagate through the indicator.

That detachment is deliberate. The indicator `1[x < R]` has gradient
zero almost everywhere and a Dirac at `x = R`; SGD on a true 0/Dirac
landscape cannot make any progress. We need a smooth replacement that
*approaches* the indicator as we want and stays differentiable
everywhere.

---

## 2. The sigmoid surrogate

The standard choice is

```
P_out^{smooth}(F, R, tau) = E_H[ sigma((R - I) / tau) ],
```

where `sigma(z) = 1 / (1 + exp(-z))` is the logistic sigmoid. Several
properties make this a comfortable training loss:

- **Recovery in the limit.** `sigma((R - I)/tau)` is `1` when
  `I < R` and `0` when `I > R` as `tau → 0`; the surrogate converges
  pointwise to the indicator. (The test suite checks this in
  [`tests/test_theoretical_validation.py::test_outage_smooth_converges_to_closed_form_as_tau_decreases`](../tests/test_theoretical_validation.py).)
- **Smoothness.** For any `tau > 0`, the surrogate is `C^∞` in `F` —
  back-prop flows cleanly through every link of the chain.
- **Bounded output.** `P_out^{smooth} ∈ (0, 1)` always; no clipping or
  rescaling needed.

`fading_dag.outage.outage_probability_smooth(I_samples, R, tau)`
returns this scalar.

---

## 3. A first outage-minimization loop

The closure is identical to Tutorial 3's except that the aggregation
step calls `outage_probability_smooth` instead of `.mean()`, and the
optimizer is `sgd_descent` instead of `sgd_ascent`:

```python
import torch
from fading_dag import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
    outage_probability, outage_probability_smooth,
    project_frobenius_ball,
    samplers,
    sgd_descent,
)

torch.manual_seed(11)
d, sigma2, P, R, tau, B = 2, 1.0, 5.0, 1.5, 0.3, 128
DTYPE = torch.complex128

F = (0.3 * torch.randn(d, d, dtype=DTYPE)).requires_grad_(True)
H_sampler = samplers.rayleigh((d, d))
Sigma_X = torch.eye(d, dtype=DTYPE)
Sigma_Z = sigma2 * torch.eye(d, dtype=DTYPE)
edge_mats = {(1, 0): (H_sampler, F)}

def smooth_outage_cost() -> torch.Tensor:
    K = compute_k_blocks_multiroot(
        num_nodes=2,
        roots=[0],
        parents={1: [0]},
        edge_mats=edge_mats,
        root_covs={0: Sigma_X},
        noise_covs={1: Sigma_Z},
        batch_size=B,
    )
    I = conditional_mutual_information_from_k(K, A=[0], B=[1]).real
    return outage_probability_smooth(I, R=R, tau=tau)

history = sgd_descent(
    smooth_outage_cost,
    [F],
    step_size=0.1,
    num_iters=500,
    projector=lambda ps: [project_frobenius_ball(p, P=P) for p in ps],
)
print(f"smooth outage: {history[0]:.4f} -> {history[-1]:.4f}")
```

`sgd_descent` shares its signature and history convention with
`sgd_ascent`; the returned history is in the *true sign* of the
cost (monotonically non-increasing on a successful descent).
Internally it negates the closure and forwards to `sgd_ascent`, then
flips the sign of the recorded values — so you read the history just
like an ascent loop, only "going down".

A typical run prints

```
smooth outage: 0.9561 -> 0.0662
```

— a ~14× reduction in 500 iterations.

---

## 4. Train smooth, evaluate raw

The surrogate is convenient for training; the *raw* indicator is the
performance measure you want to report. A clean post-training
evaluation pattern is:

```python
with torch.no_grad():
    K_eval = compute_k_blocks_multiroot(
        num_nodes=2,
        roots=[0],
        parents={1: [0]},
        edge_mats=edge_mats,
        root_covs={0: Sigma_X},
        noise_covs={1: Sigma_Z},
        batch_size=10_000,
    )
    I_eval = conditional_mutual_information_from_k(K_eval, A=[0], B=[1]).real
    raw = outage_probability(I_eval, R=R).item()
    smooth = outage_probability_smooth(I_eval, R=R, tau=tau).item()
print(f"raw outage @ B=10k       : {raw:.4f}")
print(f"smooth outage @ B=10k    : {smooth:.4f}")
print(f"raw vs smooth gap        : {abs(raw - smooth):.4f}")
```

The raw outage is the number you would put in a paper or system spec;
the smooth value is reported alongside as a sanity check. The
two should be close at small `tau` (Tutorial 4 in cmi-dag uses the
same pattern for sign-indefinite objectives).

---

## 5. Sigmoid saturation

The single most important pitfall in outage-surrogate training is
**gradient saturation**. The gradient of the surrogate with respect to
`F` flows through `sigma'((R - I) / tau)`, which equals
`sigma · (1 - sigma)`. This number is at most `0.25` (at `sigma = 0.5`,
i.e. `I = R`) and falls off exponentially as `sigma → 0` or `1`. When
every realization sits far on one side of the threshold, the gradient
*vanishes* and `F` stops moving.

Three symptoms to recognize:

- **Tiny `F` at initialization.** Every realization gives
  `I ≈ 0 ≪ R`, so `sigma ≈ 1` everywhere and the gradient is ~0. `F`
  never grows.
- **Aggressively small `tau`.** The surrogate becomes a near-step
  function. Almost every realization is far from `R` in unit of `tau`,
  so the per-realization sigmoid is ~0 or ~1 and the average gradient
  collapses.
- **High-noise channel.** With large `sigma_Z`, `I` has wide variance.
  If even the high-end realizations sit below `R`, the surrogate
  saturates near 1 from below — no gradient signal that "moving `F`
  along this direction reduces outage".

### Three remedies

1. **Initialize `F` at a moderate magnitude.** Half the Frobenius
   budget is a good default. Some realizations will then *already*
   straddle `R`, putting the sigmoid in its responsive region and
   gradient meaningfully nonzero.

   ```python
   import math
   F_init = math.sqrt(P) / 2 * torch.randn(d, d, dtype=DTYPE)
   F_init = F_init / torch.linalg.norm(F_init) * (math.sqrt(P) / 2)
   F = F_init.clone().requires_grad_(True)
   ```

2. **Use a generous `tau`, then anneal.** Start with `tau ∈ [0.3, 0.5]`
   so the sigmoid covers a wide responsive band. If a sharper
   approximation is needed, anneal `tau` down once the surrogate has
   moved into a non-saturated regime. Picking a fixed `tau ≈ 0.3` is
   often enough for the SISO / 2×2 MIMO cases in the test suite and
   examples.

3. **Use a large mini-batch.** Mini-batch noise on the surrogate
   gradient grows as `1/√B`; pushing `B` up makes the gradient step
   reliable enough to break out of nearly-saturated regions. `B = 128`
   is sometimes enough, `B = 256–1024` is safer.

The included `examples/outage_minimizing_precoder.py` illustrates all
three. The test
[`tests/test_theoretical_validation.py::test_sgd_descent_reaches_closed_form_siso_outage_minimum`](../tests/test_theoretical_validation.py)
exercises the loop on SISO Rayleigh, verifying that the *trained* raw
outage matches the closed-form optimum
`1 - exp(-(e^R - 1) / P)` within Monte Carlo tolerance.

---

## 6. Picking `R`

The target rate `R` shapes the problem more than any other knob:

- **`R` well above achievable mean MI** → outage is large and
  insensitive to `F`; surrogate is flat. Lower `R`.
- **`R` well below achievable mean MI** → outage is tiny at
  initialization; surrogate is saturated from below.
- **`R` near the ergodic capacity** → meaningful optimization
  landscape; `F` will trade off mean MI for tail behavior.

For SISO Rayleigh with budget `P`, the ergodic capacity is
`exp(1/P) · E_1(1/P)`. Picking `R` at, say, 50% of that ergodic value
is a reasonable starting point for outage minimization.

---

## 7. What is next?

- **Tutorial 5** combines everything from the first four tutorials
  into a fading multi-root MAC: per-realization conditional MI on the
  pentagon facets, ergodic rate-region comparisons, and outage
  comparisons. This is the capstone tutorial; after working through
  it you will have a complete picture of every fading-DAG idiom the
  library exposes.
