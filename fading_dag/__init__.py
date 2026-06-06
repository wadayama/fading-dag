"""fading-dag: fading-channel mutual information and SGD optimization for
linear Gaussian DAGs.

Sister library to `cmi-dag`. Extends the multi-root K-recursion + conditional
mutual information evaluator to **fading-channel** scenarios via mini-batched
Monte Carlo over channel-matrix realizations.

Each edge of the DAG is a 2-tuple ``(H_sampler, F)``: the random channel
``H`` is drawn batchwise from the sampler on each forward pass, while the
controllable factor ``F`` (precoder, relay gain, ...) is a single
deterministic tensor shared across all batch realizations. The K-recursion
is batched; gradients with respect to ``F`` are obtained by mean-aggregating
the per-realization scalar losses.

This library is fully self-contained: numerical primitives are vendored
from `cmi-dag` so there is no `cmi-dag` runtime dependency.
"""

from fading_dag import samplers  # submodule re-export
from fading_dag.information import (
    conditional_mutual_information_from_k,
    logdet_hpd,
)
from fading_dag.krecursion import (
    compute_k_blocks_multiroot,
    get_K,
    hermitianize,
)
from fading_dag.optimize import sgd_ascent, sgd_descent
from fading_dag.outage import (
    ergodic_capacity,
    outage_probability,
    outage_probability_smooth,
)
from fading_dag.projections import project_frobenius_ball, project_total_power
from fading_dag.rate_region import Summand, evaluate_rate_functions

__all__ = [
    "Summand",
    "compute_k_blocks_multiroot",
    "conditional_mutual_information_from_k",
    "ergodic_capacity",
    "evaluate_rate_functions",
    "get_K",
    "hermitianize",
    "logdet_hpd",
    "outage_probability",
    "outage_probability_smooth",
    "project_frobenius_ball",
    "project_total_power",
    "samplers",
    "sgd_ascent",
    "sgd_descent",
]
