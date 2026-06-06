# Tutorial 1 — Installation and your first per-realization MI

This first tutorial walks through installing `fading-dag` and computing
the **per-realization mutual information** of a single-input
single-output (SISO) Rayleigh fading channel — the smallest non-trivial
fading model. Per-realization MI is the building block of every
fading-channel performance measure (ergodic capacity, outage
probability, rate-region facets) computed in the rest of the tutorial
series.

By the end of this tutorial you will:

- Have a working `fading-dag` environment.
- Understand the SISO Rayleigh model as a 2-node single-root DAG with
  a `(H_sampler, F)` 2-tuple edge.
- Have computed a length-`B` vector of per-realization MIs in a single
  K-recursion forward pass.
- Have aggregated those samples into an **ergodic capacity** estimate
  and checked that it matches Telatar's closed form
  `exp(1/γ) · E_1(1/γ)` within Monte Carlo tolerance.

---

## 1. Install the library

`fading-dag` is a small Python package built on PyTorch. It has **no
runtime dependency** on its sister libraries `gaussian-dag` and
`cmi-dag`: the generic numerical primitives are vendored. Use
[`uv`](https://docs.astral.sh/uv/) to manage the virtual environment.

```bash
# Clone the repository.
git clone https://github.com/wadayama/fading-dag.git
cd fading-dag

# Install dependencies into a fresh .venv (Python >= 3.12 required).
uv sync
```

Confirm the install:

```bash
uv run pytest
```

You should see all 68 tests pass in about a second on CPU.

The example scripts also need `matplotlib`, and the
theoretical-validation tests need `scipy`. Both are optional extras:

```bash
uv sync --extra examples       # matplotlib
uv sync --extra dev            # scipy + pytest
```

---

## 2. The model

A 2×2 single-link Rayleigh MIMO channel transmits

```
   X  ──►  [ H F ]  ──►  Y = H F X + Z,
                              X ~ CN(0, I_d),
                              Z ~ CN(0, σ² I_d),
                              H ~ CN(0, I_d ⊗ I_d) drawn fresh per realization.
```

`F` is the deterministic controllable precoder; `H` is the random
channel. In DAG language this is a 2-node graph with **one root** (`X`)
and one non-root sink (`Y`):

- Node `V_0 = X` is the user-input root, with `X ~ CN(0, I_d)`.
- Node `V_1 = Y` is a non-root with parent `V_0` and edge transform
  `A_{1,0} = H F` — *random per realization*.

In `fading-dag`, the random part `H` and the deterministic part `F`
are kept **separate** in the edge specification:

```python
edge_mats = {(1, 0): (H_sampler, F)}
```

Here `H_sampler: Callable[[int], Tensor]` is a callable that, given a
batch size `B`, returns `B` independent realizations of `H` stacked
along the leading axis. `F` is a single deterministic tensor shared
across all realizations. We will pick `H_sampler = samplers.rayleigh((d, d))`,
which yields i.i.d. `CN(0, 1)` entries.

The per-realization MI is

```
I^{(b)}(X; Y) = log det(I + σ^{-2} H^{(b)} F F^H (H^{(b)})^H),
```

evaluated independently for each `b = 0, ..., B-1`. The K-recursion of
`fading-dag` produces all `B` log-determinants in a single forward pass.

---

## 3. Compute `B` per-realization MIs in one forward pass

```python
import torch
from fading_dag import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
    samplers,
)

torch.manual_seed(0)
d, sigma2, B = 2, 0.5, 1000

# Device-agnostic: same code runs on CPU or CUDA.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.complex128

# Identity precoder (no precoding yet; we'll optimize F in Tutorial 3).
F = torch.eye(d, dtype=DTYPE, device=DEVICE)
H_sampler = samplers.rayleigh((d, d), dtype=DTYPE, device=DEVICE)

# Single-root, single-sink DAG.
K = compute_k_blocks_multiroot(
    num_nodes=2,
    roots=[0],
    parents={1: [0]},
    edge_mats={(1, 0): (H_sampler, F)},
    root_covs={0: torch.eye(d, dtype=DTYPE, device=DEVICE)},
    noise_covs={1: sigma2 * torch.eye(d, dtype=DTYPE, device=DEVICE)},
    batch_size=B,
)

I_samples = conditional_mutual_information_from_k(K, A=[0], B=[1]).real
print(f"I_samples.shape  = {tuple(I_samples.shape)}")
print(f"I_samples[:5]    = {I_samples[:5].tolist()}")
print(f"E[I] estimate    = {I_samples.mean().item():.4f} nats")
print(f"sample std       = {I_samples.std().item():.4f}")
```

What just happened:

- The Rayleigh sampler was called *exactly once* with `B = 1000`, returning
  a `(1000, 2, 2)` complex tensor of i.i.d. channel realizations.
- `compute_k_blocks_multiroot` propagated the root covariance through
  the DAG with the realized edge matrices and produced the batched
  K-blocks
  `K[(0,0)]` (shape `(B, 2, 2)`, root self-cov, broadcast-expanded),
  `K[(1,0)]` (shape `(B, 2, 2)`, cross-cov), and
  `K[(1,1)]` (shape `(B, 2, 2)`, sink self-cov).
- `conditional_mutual_information_from_k(K, A=[0], B=[1])` read the
  K-blocks, formed the Schur complements `Σ_{0|1}` and `Σ_{0|()}` for
  each batch index `b`, and returned the **per-realization** log-det
  difference as a `(B,)` real tensor.

`I_samples[b]` is the mutual information you would have realized had
the channel taken its `b`-th value. `I_samples.mean()` is the
finite-`B` Monte Carlo estimate of the **ergodic capacity** `E_H[I]`.

> **A note on shape conventions.** Every K-block has the same leading
> batch axis after the recursion completes; root-root blocks (which are
> constant across realizations) share memory across the batch axis via
> `Tensor.expand`, so the uniform 3-D layout costs only views — not B-fold
> allocations. This invariant is what allows downstream code to treat
> all K-blocks uniformly.

---

## 4. Sanity check: SISO scalar case matches Telatar

Telatar's classical result for the scalar Rayleigh channel
`y = h x + z` with `h ~ CN(0, 1)`, `z ~ CN(0, σ²)`, and `E|x|² = γ` is

```
E_h[ log(1 + γ |h|²) ] = exp(1/γ) · E_1(1/γ)    (nats)
```

where `E_1(x) = ∫_x^∞ (e^{-t} / t) dt` is the exponential integral.
Reduce the model to `d = 1` and check:

```python
import math
import torch
from scipy.special import exp1                       # from `uv sync --extra dev`
from fading_dag import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
    samplers,
)

torch.manual_seed(0)
gamma, B = 5.0, 50_000                               # SNR = 5; large batch for low MC noise
DTYPE = torch.complex128

# Σ_X = gamma * 1, σ² = 1: this places the SNR in the input covariance,
# leaving F = 1 unchanged.
F = torch.eye(1, dtype=DTYPE)
H_sampler = samplers.rayleigh((1, 1))

K = compute_k_blocks_multiroot(
    num_nodes=2,
    roots=[0],
    parents={1: [0]},
    edge_mats={(1, 0): (H_sampler, F)},
    root_covs={0: torch.tensor([[complex(gamma, 0.0)]], dtype=DTYPE)},
    noise_covs={1: torch.tensor([[complex(1.0, 0.0)]], dtype=DTYPE)},
    batch_size=B,
)
I_samples = conditional_mutual_information_from_k(K, A=[0], B=[1]).real

empirical   = I_samples.mean().item()
theoretical = math.exp(1.0 / gamma) * float(exp1(1.0 / gamma))
print(f"library    : E[I] estimate  = {empirical:.4f}")
print(f"closed form: E[I]           = {theoretical:.4f}")
print(f"absolute gap                = {abs(empirical - theoretical):.4f}")
```

The two numbers should agree within Monte Carlo tolerance (the 4-σ
standard error at `B = 50 000` is about 0.05 nats). The test suite
checks this and the analogous identities for SISO outage probability,
batched `logdet_hpd`, and the constant-channel limit in
[`tests/test_theoretical_validation.py`](../tests/test_theoretical_validation.py).

> **Why is the SNR in the root covariance and not in F?** The model
> places `γ = E|x|²` in the input variance; `F = 1` is the trivial
> precoder. We could equivalently fold `γ` into `F` by setting
> `F = √γ · 1` and `Σ_X = 1`. Both give the same `H F X` distribution
> and the same MI. We will explore the precoder-as-design-variable
> formulation starting in Tutorial 3.

---

## 5. What is next?

- **Tutorial 2** opens up the `(H_sampler, F)` edge specification:
  why the 2-tuple, when each component matters, all five built-in
  sampler factories (Rayleigh, Ricean, Kronecker, scaled-Rayleigh,
  constant), how to write a custom sampler, and the **sampler-once
  invariant** that the K-recursion relies on.
- **Tutorial 3** makes the precoder `F` *trainable* and runs projected
  stochastic gradient **ascent** on the ergodic-capacity surrogate.
- **Tutorial 4** introduces the outage probability, its non-differentiable
  raw form, the sigmoid surrogate that fixes that, and the `sgd_descent`
  companion — together with the *sigmoid saturation* pitfall that
  every fading-channel optimization eventually runs into.
- **Tutorial 5** is the capstone: a 2-user fading MAC with two
  trainable precoders, conditional MI on each pentagon facet
  *per realization*, and a worked comparison of ergodic vs. outage
  rate regions.
