# Tutorial 3 — Ergodic capacity maximization with `sgd_ascent`

Tutorials 1–2 set up a fading channel and computed per-realization MI
samples; the precoder `F` was a *fixed* matrix. This tutorial promotes
`F` to a **trainable** tensor and runs projected stochastic gradient
ascent on the ergodic-capacity surrogate

```
J(F) = E_H[ I(X; H F X + Z) ],
```

estimated at every iteration by a fresh `B`-sample mini-batch. The
result is the basic "fading-channel precoder design" pattern; the
later tutorials build on it.

By the end of this tutorial you will:

- Understand the **mini-batch surrogate** for ergodic capacity and why
  it is the natural SGD objective.
- Have a working SGD-ascent loop for a 2×2 Rayleigh MIMO precoder.
- Understand why SGD histories are non-monotone (mini-batch noise) and
  how to read them anyway.
- Be comfortable using `project_frobenius_ball` and (later)
  `project_total_power` as projectors for `sgd_ascent`.

---

## 1. The mini-batch surrogate

For a fixed batch `H_1, …, H_B` of channel realizations, the Monte
Carlo estimate of the ergodic capacity is

```
Ĵ_B(F) = (1/B) · Σ_{b=1}^{B} I(X; H_b F X + Z).
```

It is differentiable in `F` whenever `I` is — which is always, given
the log-det chain `Σ_Y(F) → log det → ...`. SGD on `Ĵ_B(F)` is then a
straight application of projected gradient ascent on a stochastic
surrogate of `J(F)`:

- The K-recursion forward pass computes `I(X; H_b F X + Z)` for every
  `b` in one call.
- `.mean()` aggregates the `(B,)` tensor into a single differentiable
  scalar.
- `loss.backward()` (folded inside `sgd_ascent`) flows the chain-rule
  gradient back into `F`.
- `sgd_ascent` adds `step_size · ∂Ĵ_B / ∂F*` to `F`, then projects.

Because the K-recursion re-samples `H` on every closure call, every
iteration's gradient is based on a *fresh* batch — the textbook SGD
setup.

---

## 2. Build the closure

```python
import torch
from fading_dag import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
    project_frobenius_ball,
    samplers,
    sgd_ascent,
)

torch.manual_seed(0)
d, sigma2, P, B = 2, 0.5, 4.0, 64
DTYPE = torch.complex128

# Trainable precoder: complex, requires_grad, initialised at moderate magnitude.
F = (0.2 * torch.randn(d, d, dtype=DTYPE)).requires_grad_(True)
H_sampler = samplers.rayleigh((d, d))

Sigma_X = torch.eye(d, dtype=DTYPE)
Sigma_Z = sigma2 * torch.eye(d, dtype=DTYPE)
edge_mats = {(1, 0): (H_sampler, F)}

def ergodic_capacity_surrogate() -> torch.Tensor:
    """Closure called once per SGD iteration; re-samples H internally."""
    K = compute_k_blocks_multiroot(
        num_nodes=2,
        roots=[0],
        parents={1: [0]},
        edge_mats=edge_mats,
        root_covs={0: Sigma_X},
        noise_covs={1: Sigma_Z},
        batch_size=B,
    )
    I_samples = conditional_mutual_information_from_k(K, A=[0], B=[1]).real
    return I_samples.mean()                      # surrogate Ĵ_B(F)
```

Three details worth flagging:

- **`F` is a `leaf` tensor** with `requires_grad=True`. The K-recursion
  weaves it into the autograd graph via `H @ F` (and downstream
  log-det); `sgd_ascent` will see exactly one `Tensor.grad` to update.
- **The closure takes no arguments.** This matches the cmi-dag
  `pga_ascent` API verbatim. The mini-batch size `B` is captured by
  closure; substitute a per-iteration override (e.g.
  `B = B_warmup_schedule(t)`) at the cost of writing
  `compute_mi(t=...)`.
- **The closure does not zero gradients.** That is the optimizer's
  job (`sgd_ascent` zeroes `F.grad` before every `backward()`).

---

## 3. The projector

The Frobenius-ball constraint `||F||_F^2 ≤ P` is enforced by
`project_frobenius_ball`. It is a *functional* projector: it returns a
new tensor of the same shape rather than mutating in place.

```python
def projector(params):
    return [project_frobenius_ball(p, P=P) for p in params]
```

`sgd_ascent` accepts either functional or in-place projectors; the
choice is yours and is documented in
[`fading_dag.optimize.sgd_ascent`](../fading_dag/optimize.py). For a
list of precoders that share a single total budget, use
`project_total_power(params, P)` instead.

---

## 4. Run the loop

```python
history = sgd_ascent(
    ergodic_capacity_surrogate,
    [F],
    step_size=0.05,
    num_iters=300,
    projector=projector,
)

print(f"Iter   0  E[I] ≈ {history[0]:.3f} nats")
print(f"Iter 300  E[I] ≈ {history[-1]:.3f} nats")
print(f"||F||_F   = {torch.linalg.norm(F.detach()).item():.3f}   (budget √P = {P**0.5:.3f})")
```

A typical run prints something like

```
Iter   0  E[I] ≈ 0.513 nats
Iter 300  E[I] ≈ 3.520 nats
||F||_F   = 2.000   (budget √P = 2.000)
```

`E[I]` rises from ~0.5 to ~3.5 nats, `F` saturates the Frobenius-ball
boundary. The polished version with figure output is in
[`examples/ergodic_mimo_precoder.py`](../examples/ergodic_mimo_precoder.py).

---

## 5. Read the history

```python
print(history[:10])         # first 10 iterations of Ĵ_B(F^{(t)})
```

The history is **not monotone**. Each entry is the surrogate evaluated
at the pre-update parameter state on a *new* mini-batch of `H`
realizations, so the value bounces by ~`std(I) / sqrt(B)` from one
iteration to the next even if `F` is improving on average. Three
strategies for reading it:

- **Trailing-window mean.** Average a window of recent values (50, 100
  iterations) and look at how that smoothed sequence behaves. This is
  the de facto "convergence trace" in fading-channel SGD.
- **Fixed-batch evaluation.** Periodically, call
  `ergodic_capacity_surrogate()` with a much larger batch (e.g.
  `B_eval = 1024` or `10 000`) inside `torch.no_grad()`. That gives a
  *low-noise* estimate of the true ergodic capacity at the current
  `F`, decoupled from the training mini-batch size.
- **Closed-form comparison.** When a closed form exists (SISO
  Rayleigh, low-`d` MIMO with i.i.d. Rayleigh), compare the trained
  ergodic estimate against it; the gap should be Monte-Carlo
  tolerance. The library does exactly this in
  [`tests/test_theoretical_validation.py::test_siso_rayleigh_ergodic_capacity_matches_telatar`](../tests/test_theoretical_validation.py).

---

## 6. A larger evaluation batch

A clean way to detach evaluation from training:

```python
with torch.no_grad():
    K_eval = compute_k_blocks_multiroot(
        num_nodes=2,
        roots=[0],
        parents={1: [0]},
        edge_mats=edge_mats,                       # F is still the trained leaf
        root_covs={0: Sigma_X},
        noise_covs={1: Sigma_Z},
        batch_size=10_000,
    )
    I_eval = conditional_mutual_information_from_k(K_eval, A=[0], B=[1]).real
print(f"ergodic capacity @ B=10 000: {I_eval.mean().item():.4f} ± {I_eval.std().item() / 10_000**0.5:.4f}")
```

This evaluation pattern is the basis for every comparison in Tutorials 4
and 5 (where we will compare the smooth-surrogate training value
against the raw outage probability on a large evaluation batch).

---

## 7. Hyperparameters that matter

For a typical 2-to-4-antenna fading-channel precoder problem:

| Knob | Reasonable range | Notes |
| --- | --- | --- |
| `step_size` | 0.01–0.2 | Larger ⇒ faster ascent but bigger mini-batch noise per step. |
| `B` (`batch_size`) | 32–256 | Larger ⇒ smaller mini-batch noise; cost per iter scales linearly. |
| `num_iters` | 200–2 000 | Watch the trailing-window mean and stop when it plateaus. |
| `P` (budget) | application-specific | The budget shapes the achievable ergodic capacity ceiling. |
| `init magnitude` | half the budget | Avoid the trivial `F ≈ 0` saddle; see Tutorial 4 for the related sigmoid-saturation issue under outage objectives. |

For the SISO Rayleigh case the optimal `F` is the *full-power*
precoder (`|F|^2 = P` in any direction); the achievable ergodic
capacity is the Telatar closed form
`exp(σ²/P) · E_1(σ²/P)`. For MIMO, the optimal `F` whitens the input
covariance to track the dominant eigenmodes of `E[H^H H]`; with i.i.d.
Rayleigh, this reduces to the uniform-power waterfilling solution
under unit power per antenna.

---

## 8. What is next?

- **Tutorial 4** turns `sgd_ascent` into `sgd_descent` for *outage*
  minimization, introduces the sigmoid surrogate that bridges raw
  outage and gradient flow, and walks through the sigmoid-saturation
  pitfall and its three standard remedies.
- **Tutorial 5** generalizes everything to the **multi-root** case (a
  2-user MAC), introduces the rate-function evaluator, and shows how
  to compare *ergodic* rate regions (with `E[f_T]`) against *outage*
  rate regions (with `Pr[f_T < R_T]`).
