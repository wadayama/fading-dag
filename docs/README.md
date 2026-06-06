# fading-dag tutorials

A five-part walkthrough of the library, from a first per-realization
mutual information evaluation on a SISO Rayleigh channel to the
fading multi-root MAC with rate-region constraints.

| # | Topic | File |
| --- | --- | --- |
| 1 | Installation and your first per-realization MI | [`tutorial-1-installation-and-first-per-realization-mi.md`](tutorial-1-installation-and-first-per-realization-mi.md) |
| 2 | The `(H_sampler, F)` edge specification and sampler factories | [`tutorial-2-edge-spec-and-samplers.md`](tutorial-2-edge-spec-and-samplers.md) |
| 3 | Ergodic capacity maximization with `sgd_ascent` | [`tutorial-3-ergodic-capacity-with-sgd-ascent.md`](tutorial-3-ergodic-capacity-with-sgd-ascent.md) |
| 4 | Outage probability and the sigmoid surrogate | [`tutorial-4-outage-and-sigmoid-surrogate.md`](tutorial-4-outage-and-sigmoid-surrogate.md) |
| 5 | Fading MAC and rate functions | [`tutorial-5-fading-mac-and-rate-functions.md`](tutorial-5-fading-mac-and-rate-functions.md) |

Each tutorial is self-contained and includes runnable code snippets.
The scripts under [`../examples/`](../examples/) accompany Tutorials 3–4
as the polished end-to-end versions with figure output; reading them
is optional.

These tutorials are self-contained — `fading-dag` has no
`gaussian-dag` or `cmi-dag` runtime dependency. That said, the
deterministic-channel background (single-root K-recursion, conditional
MI, projected gradient ascent / descent, rate-function evaluator) is
introduced more gently in the parent libraries
[`gaussian-dag`](https://github.com/wadayama/gaussian-dag) (single-pair
MI) and
[`cmi-dag`](https://github.com/wadayama/cmi-dag) (multi-root, conditional
MI, rate regions). Working through those tutorial series first will
make this one easier, but is not required.

## Reading order

- **Newcomers:** read the five tutorials in order. Each tutorial only
  uses concepts introduced in the earlier ones; by the end of
  Tutorial 5 you will have seen every public symbol in the library.
- **Already familiar with `cmi-dag`:** Tutorials 1 and 5 are the most
  important — they cover the *fading-specific* batched API and the
  ergodic vs. outage rate-region comparison. Tutorial 2 has the
  sampler reference; skim it as needed. Tutorials 3 and 4 are short
  reformulations of `cmi-dag`'s `pga_ascent` / `pga_descent` material
  for the stochastic case.
- **Looking for a specific recipe:** the public API table and
  conventions section of the top-level [README](../README.md) give a
  short index of every symbol with its signature, and Tutorial 5's
  closing "where to go from here" section catalogs the
  theoretical-validation tests that pin down each numerical
  invariant.
