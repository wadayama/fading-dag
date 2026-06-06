# Tutorial 2 — The `(H_sampler, F)` edge specification and sampler factories

Tutorial 1 used `edge_mats[(j, i)] = (H_sampler, F)` without explaining
*why* the edge is a 2-tuple or what the five built-in sampler factories
do. This tutorial opens both up: the design rationale, a complete tour
of `fading_dag.samplers`, how to roll your own sampler when none of the
built-ins fit, and the **sampler-once invariant** that the K-recursion
relies on for correctness.

By the end of this tutorial you will:

- Understand why the random channel `H` and the deterministic factor
  `F` are kept structurally separate in every fading-DAG edge.
- Know which built-in sampler fits which physical scenario, and how to
  parameterize each one.
- Be able to write a custom `H_sampler` for any application your
  fading model demands (block fading, AR-1 time correlation,
  user-supplied measurement traces, ...).
- Know exactly when each edge's sampler is called per forward pass —
  the invariant your custom sampler must respect.

---

## 1. Why a 2-tuple?

In the deterministic-channel library `cmi-dag`, every edge is a single
matrix: `edge_mats[(j, i)] = A_{ji}`. The user is free to bake whatever
factorization they want into `A_{ji}` themselves
(`A_{ji} = H_j F_j`, or `A_{ji} = G_j H_j F_j`, ...).

For fading channels the random and the deterministic parts of `A_{ji}`
play *different roles*:

- **`H` is random.** It is drawn fresh every forward pass; its
  distribution carries the fading model. The library needs to call its
  sampler exactly once per K-recursion invocation, and the resulting
  batched tensor flows into the forward graph as a *leaf without
  gradient* (no gradient flows back into the channel realization).
- **`F` is deterministic.** It is shared across every realization in
  the mini-batch and across iterations of the SGD loop. It is typically
  a leaf `requires_grad=True`; the entire purpose of training is to
  push gradient signal into it.

Encoding these two roles separately in the API has three benefits:

1. **The library can dispatch sampling automatically.** Without the
   2-tuple the user would have to materialize `H @ F` themselves on
   every call; the library would lose the chance to call samplers
   exactly once per forward pass (more on this in §4 below).
2. **The fading model becomes a hot-swappable knob.** Replacing the
   sampler with a different distribution (Rayleigh → Ricean) takes one
   line — no other code changes.
3. **Custom samplers are first-class.** Block fading, AR-1 time
   correlation, user-supplied measurement traces, and any other
   "non-standard" fading model can be plugged into a fading-DAG without
   touching the library's internals (§3).

Whenever you need an edge that has no random part (a fixed cascade,
relay→destination wiring that does not fade), wrap the deterministic
matrix in `samplers.constant`; whenever you need an edge with no
controllable part, set `F = torch.eye(...)`. The 2-tuple convention is
**strict** — the K-recursion validates it on every call.

---

## 2. Tour of the built-in samplers

All factories live under `fading_dag.samplers` and return a
`Callable[[int], Tensor]` producing `(B, d_out, d_in)` complex tensors.

### `samplers.rayleigh(shape, *, dtype=torch.complex128, device=None)`

i.i.d. Rayleigh fading: each entry is an independent `CN(0, 1)` sample.
The magnitudes are Rayleigh-distributed, the phases are uniform on
`[0, 2π)`.

```python
from fading_dag import samplers
sampler = samplers.rayleigh((4, 2))         # d_out=4, d_in=2
H = sampler(8)                              # shape (8, 4, 2), complex128
```

This is the default model for rich-scattering small-scale fading.

### `samplers.ricean(shape, H_LOS, K, *, dtype=torch.complex128, device=None)`

Ricean fading: deterministic line-of-sight component combined with a
Rayleigh diffuse component,

```
H = √(K/(K+1)) · H_LOS + √(1/(K+1)) · CN(0, I_{d_out × d_in}),
```

where `K ≥ 0` is the Ricean K-factor (LoS power / scattered power) and
`H_LOS` is a fixed `(d_out, d_in)` complex tensor. `K = 0` collapses
back to Rayleigh; `K → ∞` collapses to deterministic `H_LOS`.

```python
import torch
H_LOS = torch.full((4, 2), 1.0 + 0.0j, dtype=torch.complex128)
sampler = samplers.ricean((4, 2), H_LOS, K=10.0)
```

### `samplers.kronecker(shape, R_rx, R_tx, *, dtype=torch.complex128, device=None)`

Spatially correlated Rayleigh fading via Kronecker separability,

```
H = R_rx^{1/2} · H_iid · R_tx^{1/2},
```

where `R_rx` (shape `(d_out, d_out)`) and `R_tx` (shape `(d_in, d_in)`)
are Hermitian positive-semi-definite correlation matrices and `H_iid`
is i.i.d. `CN(0, 1)`. The factory pre-caches the Cholesky factors of
`R_rx` and `R_tx` so the per-call cost is identical to plain
`rayleigh`.

```python
import torch
d_out, d_in = 4, 2
R_rx = torch.eye(d_out, dtype=torch.complex128) + 0.5 * torch.ones(d_out, d_out, dtype=torch.complex128) / d_out
R_tx = torch.eye(d_in, dtype=torch.complex128)
sampler = samplers.kronecker((d_out, d_in), R_rx, R_tx)
```

Use this when you have a separable spatial correlation model — for
example, antenna arrays with measured covariance matrices.

### `samplers.scaled_rayleigh(shape, sigma, *, dtype=torch.complex128, device=None)`

Rayleigh fading with per-entry standard deviation `sigma` (scalar or
broadcasted tensor):

```
H_{ij} = sigma · CN(0, 1).        # per entry
```

`sigma` can be a `float` (uniform scaling) or a tensor of shape
`(d_out, d_in)` (per-entry path-loss profile, for example).

```python
import torch
# Distance-dependent path loss across 4 receive antennas:
sigma_per_row = torch.tensor([1.0, 0.7, 0.5, 0.3]).unsqueeze(1)        # (4, 1)
sampler = samplers.scaled_rayleigh((4, 2), sigma_per_row)
```

### `samplers.constant(H_fixed)`

Deterministic, non-fading "sampler": every call returns the same
underlying matrix, broadcast over the batch axis via `Tensor.expand`.
Use this to combine fading and non-fading edges in the same DAG.

```python
import torch
H_fixed = torch.eye(2, dtype=torch.complex128)
sampler = samplers.constant(H_fixed)
print(sampler(5).shape)                # (5, 2, 2); all five entries equal
```

Setting `B = 1` and using `samplers.constant` everywhere makes
`fading-dag` numerically equivalent to `cmi-dag` (verified by
`tests/test_krecursion_batched.py::test_b1_constant_matches_unbatched_recursion`).

---

## 3. Writing a custom sampler

A sampler is *any* callable that takes an `int` and returns a complex
3-D tensor of the right shape; the library does not introspect it. This
covers a wide range of practical scenarios that the built-in factories
do not.

### Example: block fading

Suppose each fading realization holds for a "block" of `block_len`
consecutive batch entries:

```python
from fading_dag import samplers

def block_fading_sampler(d_out: int, d_in: int, block_len: int):
    """Each block of block_len consecutive realizations shares one channel."""
    base = samplers.rayleigh((d_out, d_in))

    def sampler(B: int):
        n_blocks = (B + block_len - 1) // block_len
        H_blocks = base(n_blocks)                              # (n_blocks, d_out, d_in)
        H = H_blocks.repeat_interleave(block_len, dim=0)       # (n_blocks * block_len, d_out, d_in)
        return H[:B]                                           # trim to exactly B

    return sampler
```

This sampler still respects the `Callable[[int], Tensor]` contract and
slots into `edge_mats` exactly like the built-ins.

### Example: user-supplied measurement trace

If you have an `(N, d_out, d_in)` tensor of measured channel
realizations and want to cycle through them:

```python
import torch

def trace_sampler(H_trace: torch.Tensor):
    """Cycle through the rows of H_trace; B may be larger than N."""
    N = H_trace.shape[0]
    cursor = [0]                                  # mutable closure state

    def sampler(B: int):
        idx = [(cursor[0] + b) % N for b in range(B)]
        cursor[0] = (cursor[0] + B) % N
        return H_trace[idx]

    return sampler
```

For a non-trivial SGD run you typically want the trace to be larger
than the mini-batch so that the optimizer sees fresh realizations on
each step.

---

## 4. The sampler-once invariant

A subtle but load-bearing property of `compute_k_blocks_multiroot` is
that **each edge's sampler is called exactly once per invocation**,
regardless of how many cross- and self-block updates need its
result. The function realizes every edge upfront:

```python
A_eff[(j, i)] = H_sampler(batch_size) @ F        # called once, stored
```

and then reuses `A_eff[(j, i)]` for every cross-block update
`K_{jk} += A_{ji} K_{ik}` and self-block update
`K_{jj} += A_{ji} K_{ii'} A_{ji'}^H`. This guarantees realization
consistency: the cross-block `K_{2,0}` and the self-block `K_{2,2}`
that includes its parent `V_0` are built from the *same* `H_{2,0}`
sample, not two independent draws.

Two practical consequences:

- **Stateful samplers see exactly one call per K-recursion forward
  pass.** A counter inside a custom sampler increments by exactly 1
  per `compute_k_blocks_multiroot(...)` call — useful both for tests
  and for stateful fading models like the trace sampler above.
- **Re-sampling is the closure's responsibility, not the sampler's.**
  The `sgd_ascent` / `sgd_descent` loops do *not* re-draw `H` between
  iterations on their own; what they do is call `compute_mi()` again on
  the next iteration, which in turn calls
  `compute_k_blocks_multiroot(..., batch_size=...)` again, which in turn
  invokes each sampler once more — yielding a fresh mini-batch every
  step. The closure is the place to put the K-recursion call (and any
  re-sampling logic you want layered on top of it).

A simple test for the invariant looks like this:

```python
counts = {0: 0}

def counting_sampler(B):
    counts[0] += 1
    return samplers.rayleigh((2, 2))(B)

K = compute_k_blocks_multiroot(
    num_nodes=2,
    roots=[0],
    parents={1: [0]},
    edge_mats={(1, 0): (counting_sampler, torch.eye(2, dtype=torch.complex128))},
    root_covs={0: torch.eye(2, dtype=torch.complex128)},
    noise_covs={1: torch.eye(2, dtype=torch.complex128)},
    batch_size=64,
)
assert counts[0] == 1       # sampler was called exactly once for this forward pass
```

The library's test suite enforces this invariant in
[`tests/test_krecursion_batched.py::test_sampler_called_exactly_once_per_edge_per_invocation`](../tests/test_krecursion_batched.py).

---

## 5. Mixing fading and non-fading edges

A common practical setup mixes fading edges (RF links) with constant
ones (digital cascades or measured matrices). `samplers.constant` is
the bridge:

```python
import torch
from fading_dag import compute_k_blocks_multiroot, samplers

d = 2
H_fade = samplers.rayleigh((d, d))             # source -> relay: Rayleigh
F_relay = torch.eye(d, dtype=torch.complex128) # relay precoder (identity here)
H_const = samplers.constant(torch.eye(d, dtype=torch.complex128))
                                               # relay -> destination: constant identity
R = torch.eye(d, dtype=torch.complex128)       # destination-side processing

K = compute_k_blocks_multiroot(
    num_nodes=3, roots=[0],
    parents={1: [0], 2: [1]},
    edge_mats={
        (1, 0): (H_fade,  F_relay),            # fading edge
        (2, 1): (H_const, R),                  # constant edge wrapped in constant sampler
    },
    root_covs={0: torch.eye(d, dtype=torch.complex128)},
    noise_covs={1: torch.eye(d, dtype=torch.complex128),
                2: torch.eye(d, dtype=torch.complex128)},
    batch_size=32,
)
```

Note that the *constant* edge still consumes a `batch_size`-dimensional
expand, so the downstream K-blocks remain uniformly 3-D and the
recursion is unaffected.

---

## 6. What is next?

- **Tutorial 3** adds gradients to `F`: ergodic capacity surrogates,
  projected SGD ascent, the Frobenius-ball projector, and what an SGD
  history looks like under mini-batch noise.
- **Tutorial 4** introduces the outage probability and its sigmoid
  surrogate — the standard tool for back-propagating through
  `Pr[I < R]`, together with its (sometimes nasty) saturation pitfall.
- **Tutorial 5** combines the multi-root MAC of `cmi-dag` Tutorial 5
  with fading sampling: per-realization rate-region constraints and
  ergodic-vs-outage rate-region comparison.
