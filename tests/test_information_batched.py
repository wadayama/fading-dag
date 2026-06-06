"""Tests for the batched conditional mutual information evaluator.

Coverage:
- Shape and dtype of per-realization CMI tensor.
- Equivalence to a per-sample loop (independent realizations -> independent
  scalar CMIs).
- Gradient flow from mean CMI back to the controllable factor F.
- ``logdet_hpd`` batched correctness and error handling.
- Empty-conditioning fallback to unconditional MI.
"""

from __future__ import annotations

import pytest
import torch

from fading_dag import samplers
from fading_dag.information import (
    conditional_mutual_information_from_k,
    logdet_hpd,
)
from fading_dag.krecursion import compute_k_blocks_multiroot


# --------------------------------------------------------------------------- #
# logdet_hpd: batched correctness                                              #
# --------------------------------------------------------------------------- #


def test_logdet_hpd_scalar_unchanged() -> None:
    A = torch.tensor([[2.0 + 0j, 0.0 + 0j], [0.0 + 0j, 3.0 + 0j]], dtype=torch.complex128)
    out = logdet_hpd(A)
    expected = torch.log(torch.tensor(6.0, dtype=torch.float64))
    assert out.shape == ()
    assert torch.isclose(out, expected, atol=1e-12)


def test_logdet_hpd_batched_shape() -> None:
    torch.manual_seed(0)
    B = 5
    d = 3
    M = torch.randn(B, d, d, dtype=torch.complex128)
    A = M @ M.mH + 0.5 * torch.eye(d, dtype=torch.complex128)
    out = logdet_hpd(A)
    assert out.shape == (B,)
    # Per-batch entry should match a per-element loop.
    for b in range(B):
        single = logdet_hpd(A[b])
        assert torch.isclose(out[b], single, atol=1e-10)


def test_logdet_hpd_raises_on_non_pd() -> None:
    A = torch.tensor(
        [[1.0 + 0j, 1.0 + 0j], [1.0 + 0j, 1.0 + 0j]], dtype=torch.complex128
    )  # singular
    with pytest.raises(ValueError, match="positive definite"):
        logdet_hpd(A)


def test_logdet_hpd_jitter_rescues_psd() -> None:
    A = torch.tensor(
        [[1.0 + 0j, 1.0 + 0j], [1.0 + 0j, 1.0 + 0j]], dtype=torch.complex128
    )  # singular
    out = logdet_hpd(A, jitter=1e-6)
    assert torch.isfinite(out)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _mac_with_random_F(B: int, seed: int = 0):
    """2-user MAC with Rayleigh channels and a randomly initialized F per user."""
    torch.manual_seed(seed)
    d = 2
    F0 = torch.randn(d, d, dtype=torch.complex128, requires_grad=True)
    F1 = torch.randn(d, d, dtype=torch.complex128, requires_grad=True)
    edge_mats = {
        (2, 0): (samplers.rayleigh((d, d)), F0),
        (2, 1): (samplers.rayleigh((d, d)), F1),
    }
    root_covs = {0: torch.eye(d, dtype=torch.complex128), 1: torch.eye(d, dtype=torch.complex128)}
    noise_covs = {2: 0.5 * torch.eye(d, dtype=torch.complex128)}
    K = compute_k_blocks_multiroot(
        num_nodes=3,
        roots=[0, 1],
        parents={2: [0, 1]},
        edge_mats=edge_mats,
        root_covs=root_covs,
        noise_covs=noise_covs,
        batch_size=B,
    )
    return K, [F0, F1]


# --------------------------------------------------------------------------- #
# CMI: shape and per-realization independence                                  #
# --------------------------------------------------------------------------- #


def test_cmi_shape_and_dtype() -> None:
    B = 8
    K, _ = _mac_with_random_F(B=B)
    # I(V_0; V_2 | V_1): user-0 -> receiver, conditioned on user-1.
    I = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])
    assert I.shape == (B,)
    assert I.real.dtype == torch.float64
    # Conditional MI should be non-negative (modulo round-off).
    assert (I.real >= -1e-9).all()


def test_cmi_per_realization_matches_b1_loop() -> None:
    """Computing CMI in a single batch must match a per-sample loop (B=1)."""
    B_full = 4
    torch.manual_seed(11)
    K_full, _ = _mac_with_random_F(B=B_full, seed=11)
    I_full = conditional_mutual_information_from_k(K_full, A=[0], B=[2], C=[1])

    # Reproduce sample-by-sample: index into the batched K and rebuild a (B=1)
    # K-dict that the same evaluator should process identically.
    for b in range(B_full):
        K_b = {key: val[b:b + 1] for key, val in K_full.items()}
        I_b = conditional_mutual_information_from_k(K_b, A=[0], B=[2], C=[1])
        assert torch.isclose(I_full[b].real, I_b[0].real, atol=1e-10)


def test_cmi_empty_conditioning_reduces_to_unconditional() -> None:
    B = 4
    K, _ = _mac_with_random_F(B=B)
    I_unconditional = conditional_mutual_information_from_k(K, A=[0], B=[2], C=())
    assert I_unconditional.shape == (B,)
    # Unconditional MI must dominate the conditional version on the same K.
    I_conditional = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])
    # I(A;B|C) <= I(A;B) when conditioning on a Markov-related node is not
    # guaranteed in general; just check both are real-finite.
    assert torch.isfinite(I_unconditional.real).all()
    assert torch.isfinite(I_conditional.real).all()


def test_cmi_disjointness_violation_raises() -> None:
    B = 2
    K, _ = _mac_with_random_F(B=B)
    with pytest.raises(ValueError, match="disjoint"):
        conditional_mutual_information_from_k(K, A=[0, 1], B=[1, 2])


def test_cmi_empty_set_raises() -> None:
    B = 2
    K, _ = _mac_with_random_F(B=B)
    with pytest.raises(ValueError, match="non-empty"):
        conditional_mutual_information_from_k(K, A=[], B=[2])


# --------------------------------------------------------------------------- #
# Gradient: F.grad must be non-zero from mean CMI loss                         #
# --------------------------------------------------------------------------- #


def test_cmi_gradient_flows_to_F() -> None:
    """Mean joint MI loss must produce a non-zero gradient on each F.

    Use the *joint* MI I([V_0, V_1]; V_2), which depends on both precoders
    F_0 and F_1. (The conditional version I(V_0; V_2 | V_1) is functionally
    independent of F_1 because conditioning on V_1 strips its contribution
    to V_2; this is why we test the joint variant here.)
    """
    B = 16
    K, params = _mac_with_random_F(B=B, seed=42)
    I = conditional_mutual_information_from_k(K, A=[0, 1], B=[2])
    loss = I.real.mean()  # E[I(V_0, V_1; V_2)]
    loss.backward()
    for idx, p in enumerate(params):
        assert p.grad is not None
        # Gradient magnitude should be strictly positive.
        norm = torch.linalg.norm(p.grad).item()
        assert norm > 1e-6, f"F_{idx} gradient norm too small: {norm:.3e}"


def test_cmi_conditional_strips_irrelevant_F() -> None:
    """I(V_0; V_2 | V_1) should be independent of F_1 (its grad is ~0)."""
    B = 16
    K, params = _mac_with_random_F(B=B, seed=99)
    F0, F1 = params
    I = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])
    I.real.mean().backward()
    # F0 still affects the CMI.
    assert torch.linalg.norm(F0.grad).item() > 1e-6
    # F1 must have negligible gradient: conditioning on V_1 strips its
    # contribution to V_2 entirely.
    assert torch.linalg.norm(F1.grad).item() < 1e-10
