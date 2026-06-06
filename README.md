# fading-dag

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.12-blue.svg)](https://www.python.org/)

Fading-channel mutual-information evaluation and SGD-based optimization for
linear Gaussian directed acyclic graphs (DAGs), via mini-batched Monte Carlo
over channel-matrix realizations. Sister library to
[`cmi-dag`](https://github.com/wadayama/cmi-dag), extending the multi-root
K-recursion + conditional MI evaluator to the **fading** setting.

```
E[I(V_A; V_B | V_C)]      via mini-batch mean
Pr[Σ α · I(V_A;V_B|V_C) < R]  via empirical CDF (with smooth surrogate for gradients)
```

Each edge of the DAG is represented as a `(H_sampler, F)` 2-tuple:

- `H_sampler: Callable[[int], Tensor]` — a callable that, given a batch
  size `B`, returns `B` independent realizations of the channel matrix.
  Built-in factories are provided for common fading models (Rayleigh,
  Ricean, Kronecker-correlated, scaled-Rayleigh, deterministic).
- `F: Tensor` — the deterministic controllable factor (precoder, relay
  gain, etc.), `requires_grad=True`. Single (no batch dimension).

The K-recursion is batched over the channel realizations; gradients with
respect to `F` are obtained by mean-aggregating the per-realization scalar
losses. The same projector toolbox (Frobenius ball, total-power) used by
`cmi-dag` plugs into the SGD loop unchanged.

## Install

```bash
uv pip install -e .                  # core library
uv pip install -e ".[examples]"      # + matplotlib for the example scripts
uv pip install -e ".[dev]"           # + pytest, scipy for the test suite
```

Or with plain `pip`:

```bash
pip install -e ".[examples,dev]"
```

Python `>= 3.12` and PyTorch `>= 2.12` are required.

## Quick Start

Maximize the ergodic capacity of a 2×2 Rayleigh MIMO precoder under a
Frobenius power budget:

```python
import torch

from fading_dag import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
    project_frobenius_ball,
    samplers,
    sgd_ascent,
)

d = 2
sigma2 = 0.5
P = 4.0  # Frobenius power budget on F.

# Controllable precoder F: the design variable.
F = (0.2 * torch.randn(d, d, dtype=torch.complex128)).requires_grad_(True)

# Edge specification: (channel sampler, deterministic factor).
edge_mats = {
    (1, 0): (samplers.rayleigh((d, d)), F),
}

def ergodic_capacity_surrogate():
    """Mini-batched Monte Carlo surrogate for E[I(X; HFX + Z)]."""
    K = compute_k_blocks_multiroot(
        num_nodes=2,
        roots=[0],
        parents={1: [0]},
        edge_mats=edge_mats,
        root_covs={0: torch.eye(d, dtype=torch.complex128)},
        noise_covs={1: sigma2 * torch.eye(d, dtype=torch.complex128)},
        batch_size=64,
    )
    I = conditional_mutual_information_from_k(K, A=[0], B=[1]).real
    return I.mean()                  # E[I] over the mini-batch

def projector(params):
    return [project_frobenius_ball(p, P=P) for p in params]

history = sgd_ascent(
    ergodic_capacity_surrogate,
    [F],
    step_size=0.05,
    num_iters=300,
    projector=projector,
)

print(f"Iter 0   E[I] = {history[0]:.3f} nats")
print(f"Iter 300 E[I] = {history[-1]:.3f} nats")
print(f"||F||_F  = {torch.linalg.norm(F.detach()).item():.3f}   (budget = {P**0.5:.3f})")
```

Swap `sgd_ascent` for `sgd_descent` and the closure for
`outage_probability_smooth(I, R, tau)` to minimize outage probability;
see `examples/outage_minimizing_precoder.py` for a worked instance and the
[sigmoid saturation caveat](#caveat-sigmoid-saturation-when-training-with-outage_probability_smooth)
below.

## Modules

- `fading_dag.krecursion` — batched multi-root K-recursion.
- `fading_dag.information` — batched `logdet_hpd` and per-realization
  conditional mutual information evaluator.
- `fading_dag.samplers` — channel-matrix sampler factories.
- `fading_dag.outage` — outage probability (raw indicator + sigmoid
  surrogate) and ergodic-capacity helpers.
- `fading_dag.rate_region` — per-realization rate-function evaluator
  (rate-region constraints).
- `fading_dag.optimize` — `sgd_ascent` / `sgd_descent` mirroring
  `cmi-dag`'s `pga_ascent` / `pga_descent` signature.
- `fading_dag.projections` — vendored Euclidean projections.

## Caveat: sigmoid saturation when training with `outage_probability_smooth`

The smooth surrogate

```
out_smooth(I, R, tau) = E[ sigma((R - I) / tau) ]
```

is the standard tool for back-propagating through outage probability, but
its gradient with respect to `F` flows through `sigma'((R - I)/tau)`, which
**vanishes whenever every channel realization sits far on one side of the
threshold** (saturation regime). Two practical symptoms:

- ``F`` is initialized at a very small magnitude → every realization gives
  `I ≈ 0 ≪ R` → `sigma((R - I)/tau) ≈ 1` → `sigma'(.) ≈ 0` → gradient
  vanishes and `F` never grows.
- `tau` is set very small from the start → the surrogate is essentially a
  step function; any realization away from `R` contributes zero gradient.

Three remedies that work well in our experiments:

1. **Initialize ``F`` at a moderate magnitude**, e.g. half the Frobenius
   budget, so that some realizations already straddle the threshold and
   the sigmoid is firmly in its responsive region.
2. **Start with a generous `tau` (e.g. 0.3–0.5) and anneal it down** during
   training (or pick a fixed `tau` that yields a non-trivial initial
   surrogate value). The ``test_theoretical_validation`` suite uses
   ``tau=0.3`` precisely for this reason.
3. **Use a large mini-batch** (256–1024). Mini-batch noise on the surrogate
   gradient grows as `1/sqrt(B)`; pushing `B` up makes the gradient step
   reliable enough to break out of nearly-saturated regions.

The included ``examples/outage_minimizing_precoder.py`` illustrates all
three: ``F`` is initialized at a moderate magnitude, ``tau=0.3``, and
``batch_size=128``. With those choices the smooth outage drops by more than
an order of magnitude in 500 iterations and ``F`` saturates the budget
boundary. The companion test
``tests/test_theoretical_validation.py::test_sgd_descent_reaches_closed_form_siso_outage_minimum``
verifies that the trained SGD optimum matches the closed-form Rayleigh-SISO
optimum
`1 - exp(-(e^R - 1) / P)`
within Monte Carlo tolerance.

## Examples

Two end-to-end scripts live in `examples/`. Curated reference output
figures (from a known-good run) are committed under `docs/figures/`;
re-running the scripts regenerates fresh PNGs next to the scripts
themselves (these regenerated copies are gitignored).

```bash
uv run examples/ergodic_mimo_precoder.py
uv run examples/outage_minimizing_precoder.py
```

### `examples/ergodic_mimo_precoder.py`

2×2 Rayleigh MIMO. Maximizes the ergodic capacity
`E_H[I(X; H F X + Z)]` over the precoder `F` under `||F||_F^2 <= P`
via projected SGD ascent. With the default settings `E[I]` rises from
~0.5 nats to ~3.5 nats in 300 iterations and `F` saturates the budget
boundary.

![Ergodic capacity trajectory](docs/figures/ergodic_mimo_precoder.png)

### `examples/outage_minimizing_precoder.py`

Same 2×2 Rayleigh DAG. Minimizes the smooth outage surrogate
`E[sigma((R - I)/tau)]` at a target rate `R` via projected SGD
descent, then evaluates the raw indicator outage on a much larger
batch to confirm the surrogate's gradient direction transfers to the
true objective. With the default settings the smooth outage drops by
more than 15× (~0.96 → ~0.07; raw indicator ~0.03 on a 10k-sample
evaluation batch) and `F` saturates the budget boundary.

![Outage minimization trajectory](docs/figures/outage_minimizing_precoder.png)

## Tests

```bash
uv run pytest tests/                                                  # 68 tests
uv run pytest tests/test_theoretical_validation.py -v                 # closed-form checks
```

The `tests/test_theoretical_validation.py` suite includes Telatar's
exponential-integral formula for SISO Rayleigh ergodic capacity, the
closed-form SISO Rayleigh outage `1 - exp(-(e^R - 1) / gamma)`, an
exact match between `samplers.constant` and the deterministic log-det
MI, and an SGD-convergence test that verifies the trained outage
matches the closed-form Rayleigh-SISO optimum within Monte Carlo
tolerance.

## Standalone

This library is fully self-contained: numerical primitives are vendored
from `cmi-dag`. It has no runtime dependency on either `gaussian-dag` or
`cmi-dag` — only PyTorch and NumPy.

## Citation

This library is research infrastructure; the companion manuscript is in
preparation. A citation block will be added here once the paper is
posted.

## License

MIT.
