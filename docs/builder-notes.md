# Builder notes — fading-dag

Library-specific decisions for the named-node DAG builder, per the template in
`builder_implementation.md` §13. The shared policy lives in that document; only
the choices it delegates to the implementer are recorded here.

- **Conforms to builder_implementation.md spec version:** 0.2
- **Profiles implemented:** `conditional` (CMI with a conditioning set),
  `multiroot` (more than one source), and `stochastic/batch` (mini-batched
  Monte Carlo over channel realizations). The `correlated-roots` extension is
  **not** implemented (the fading core has no `cross_root_covs`).
- **Query method(s):**
  - `cmi(A, B, C=(), *, batch_size, bind=None, jitter=0.0)` — per-realization
    `I(V_A; V_B | V_C)`, a real tensor of shape `(B,)`.
  - `ergodic_capacity(A, B, C=(), *, batch_size, bind=None, jitter=0.0)` — the
    mini-batch mean of `cmi` (a differentiable scalar; lowers to the core's
    `ergodic_capacity`).
  - `cov(node, *, batch_size, bind=None)` — the batched self-covariance block
    `Σ_node = K_{node,node}`, shape `(B, d, d)`.
  - Returns whatever the core returns. Outage (`outage_probability` /
    `outage_probability_smooth`) is intentionally left to the functional API;
    users aggregate `cmi(...)` themselves.
- **Class name:** `GaussianDAG` — matches the worked example in spec §3 and the
  multiroot illustration in §7, and is the same class name used by the
  `gaussian-dag` and `cmi-dag` builders, for maximal cross-library
  recognizability. No collision: the name did not exist in the repo, and it is
  added to `fading_dag/__init__.__all__` alongside (never replacing) the
  existing 15 public symbols.
- **Matrix / edge input:** both **by name (string)** and **as a concrete
  object**. A concrete object is used as-is; a name is resolved at query time
  via a `bind={name: object}` mapping. An unbound name raises `ValueError` —
  data is never fabricated (spec §8). The fading-specific part (spec §7,
  binding layer):
  - each **edge** value is an `EdgeSpec = (H_sampler, F)` 2-tuple (not a plain
    matrix): `H_sampler(batch_size)` draws a `(B, d_j, d_mid)` mini-batch and
    `F` is the shared deterministic factor;
  - `cov` / `noise` are unbatched `(d, d)` tensors;
  - `_resolve` passes through both tensors and tuples; only strings are bound.
- **`batch_size` handling:** a **required keyword on every query**. The builder
  is stateless w.r.t. the batch: each query re-runs `compute_k_blocks_multiroot`
  and therefore re-samples a fresh mini-batch, matching the core's
  sampler-once-per-forward invariant and the SGD closure convention
  (`compute_mi = lambda: dag.ergodic_capacity(..., batch_size=B)`).
- **Matrix conventions:** `complex128` standard; tensors keep their own
  dtype/device. Edge keys lower to `(j, i)` for `i → j` with `i < j`; roots are
  the prefix `{0, …, K-1}`.
- **Module / namespace:** `fading_dag/builder.py`, re-exported from
  `fading_dag/__init__.py`.
- **Canonical index (spec §12):** stable topological sort — Kahn's algorithm
  with a FIFO queue, seeding and tie-breaking by build/call order. Only sources
  have in-degree 0, so they receive the contiguous prefix `0, …, K-1`, which is
  exactly the multi-root core's root convention. Exposed for structural tests
  via the internal `_lower_structure()`.
- **Deliberate divergences from the recommended idioms (§5) and why:**
  - **`add_source` keeps the spec name even though the core says "root".** The
    spec (§3, §4.3) treats "source" and "root" as synonyms and fixes
    `add_source` for cross-library recognizability.
  - **Edge value is an `EdgeSpec` tuple, not a matrix; queries carry
    `batch_size`; quantities are `(B,)`.** These are the §7 stochastic/batch
    binding-layer differences — the structural `add_source`/`add_node`/query
    shape is unchanged from the deterministic siblings.
  - **No `add_root_correlation`.** The fading core models independent roots
    only; correlated sources are out of scope here (they live in `cmi-dag`).
- **Unsupported / invalid constructs and the errors raised:**
  - An `add_node` referencing an undeclared parent (also catches self-loops) →
    `ValueError` ("Unknown parent").
  - A parentless `add_node` → `ValueError` (a source uses `add_source`).
  - A duplicate node name → `ValueError` ("Duplicate").
  - An unbound matrix/edge name at query time → `ValueError` ("not bound").
  - `cmi` with non-disjoint or empty `A`/`B` → `ValueError` (enforced by the
    core).
  - A query on a sources-only DAG (no non-root node) → `ValueError`
    ("num_roots …"), surfaced by the core's `num_roots < num_nodes` check.

## Structural-conformance vectors (spec §12)

`chain`, the `two-source (MAC-like)` graph, and `diamond` are all expressible
and verified by `tests/test_builder.py::test_structure_chain` /
`test_structure_two_source_mac` / `test_structure_diamond`. Numeric round-trip
equality (against the functional core) uses the deterministic
`samplers.constant` sampler, because a fading sampler re-draws on each forward
pass; Monte-Carlo re-sampling is checked separately with `samplers.rayleigh`.
