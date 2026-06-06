# fading-dag examples

End-to-end demonstrations of mini-batched Monte Carlo MI / outage
optimization on linear Gaussian DAGs.

## Running

These examples require the `examples` extras (matplotlib):

```bash
uv pip install -e ".[examples]"
uv run examples/ergodic_mimo_precoder.py
uv run examples/outage_minimizing_relay.py
```

## Scripts

### `ergodic_mimo_precoder.py`

2x2 Rayleigh MIMO. Optimizes a precoder `F` under `||F||_F^2 <= P` to
maximize the ergodic capacity `E_H[I(X; H F X + Z)]` via projected SGD
ascent. Saves a trajectory plot of the trailing-50 window mean MI.

### `outage_minimizing_precoder.py`

2x2 Rayleigh MIMO. Minimizes the **smooth outage probability**
`E[sigma((R - I)/tau)]` at a target rate by SGD-descent on the precoder
`F` under `||F||_F^2 <= P`, then evaluates the raw indicator probability
on a much larger batch to confirm the surrogate's gradient direction
transfers to the true objective. With the default settings the outage
drops by more than 15x (~0.95 -> ~0.07 smooth, ~0.03 raw on a 10k-sample
evaluation batch) and `F` saturates the Frobenius-ball boundary.
