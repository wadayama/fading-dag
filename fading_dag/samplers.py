"""Channel-matrix sampler factories for fading-channel DAG simulations.

Each factory in this module returns a callable of the form

    sampler: int -> torch.Tensor

which, given a batch size ``B``, produces a complex tensor of shape
``(B, d_out, d_in)`` representing ``B`` independent realizations of a
channel matrix ``H``.

The library is agnostic to the fading distribution: the K-recursion
(`fading_dag.krecursion.compute_k_blocks_multiroot`) calls the sampler once
per forward pass and threads the result through the batched matrix algebra.
Users can drop in their own custom sampler by writing any compatible
callable; the factories below are conveniences for common parametric
fading models.

All samplers default to ``torch.complex128`` and ``device='cpu'``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

# Sampler callable type alias.
Sampler = Callable[[int], torch.Tensor]


def _validate_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if not (isinstance(shape, tuple) and len(shape) == 2):
        raise ValueError(f"shape must be a 2-tuple (d_out, d_in); got {shape!r}")
    d_out, d_in = shape
    if d_out <= 0 or d_in <= 0:
        raise ValueError(f"shape entries must be positive; got {shape!r}")
    return d_out, d_in


def _randn_complex(
    batch_size: int,
    d_out: int,
    d_in: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    """Draw a batched complex Gaussian tensor of shape (B, d_out, d_in).

    Each entry is independently CN(0, 1):
    real and imaginary parts are independent N(0, 1/2), so |entry|^2 has mean 1.
    """
    if dtype == torch.complex128:
        real_dtype = torch.float64
    elif dtype == torch.complex64:
        real_dtype = torch.float32
    else:
        raise ValueError(
            f"dtype must be torch.complex64 or torch.complex128; got {dtype!r}"
        )
    half = torch.tensor(0.5, dtype=real_dtype, device=device).sqrt()
    real = torch.randn(batch_size, d_out, d_in, dtype=real_dtype, device=device) * half
    imag = torch.randn(batch_size, d_out, d_in, dtype=real_dtype, device=device) * half
    return torch.complex(real, imag)


def rayleigh(
    shape: tuple[int, int],
    *,
    dtype: torch.dtype = torch.complex128,
    device: torch.device | str | None = None,
) -> Sampler:
    """Return a Rayleigh fading sampler.

    Each entry of the returned matrix is an independent CN(0, 1) sample
    (the real and imaginary parts are independent N(0, 1/2), so each entry
    has unit variance). The magnitude follows a Rayleigh distribution.

    Args:
        shape: ``(d_out, d_in)`` of the channel matrix to be sampled.
        dtype: complex dtype.
        device: torch device on which to allocate samples.

    Returns:
        Callable ``sampler(B) -> (B, d_out, d_in)`` tensor.
    """
    d_out, d_in = _validate_shape(shape)

    def sampler(batch_size: int) -> torch.Tensor:
        return _randn_complex(batch_size, d_out, d_in, dtype=dtype, device=device)

    return sampler


def ricean(
    shape: tuple[int, int],
    H_LOS: torch.Tensor,
    K: float,
    *,
    dtype: torch.dtype = torch.complex128,
    device: torch.device | str | None = None,
) -> Sampler:
    """Return a Ricean fading sampler.

    Each realization is

        H = sqrt(K / (K + 1)) * H_LOS + sqrt(1 / (K + 1)) * H_rayleigh,

    where ``H_LOS`` is a deterministic line-of-sight matrix and
    ``H_rayleigh`` is an i.i.d. CN(0, 1) draw. ``K`` is the Ricean K-factor
    (LoS power / scattered power); ``K = 0`` recovers pure Rayleigh.

    Args:
        shape: ``(d_out, d_in)``; must equal ``H_LOS.shape``.
        H_LOS: line-of-sight matrix, shape ``(d_out, d_in)``, complex.
        K: non-negative Ricean K-factor.
        dtype: complex dtype (also used to cast H_LOS).
        device: torch device.

    Returns:
        Callable ``sampler(B) -> (B, d_out, d_in)`` tensor.
    """
    d_out, d_in = _validate_shape(shape)
    if H_LOS.shape != (d_out, d_in):
        raise ValueError(
            f"H_LOS shape {tuple(H_LOS.shape)} does not match shape {shape}"
        )
    if K < 0:
        raise ValueError(f"K-factor must be non-negative; got {K}")
    H_LOS_cast = H_LOS.to(dtype=dtype, device=device)
    a_los = float((K / (K + 1.0)) ** 0.5)
    a_ray = float((1.0 / (K + 1.0)) ** 0.5)

    def sampler(batch_size: int) -> torch.Tensor:
        h_ray = _randn_complex(batch_size, d_out, d_in, dtype=dtype, device=device)
        # Broadcast H_LOS over the batch axis.
        return a_los * H_LOS_cast.unsqueeze(0) + a_ray * h_ray

    return sampler


def kronecker(
    shape: tuple[int, int],
    R_rx: torch.Tensor,
    R_tx: torch.Tensor,
    *,
    dtype: torch.dtype = torch.complex128,
    device: torch.device | str | None = None,
) -> Sampler:
    """Return a Kronecker-correlated Rayleigh fading sampler.

    Each realization is

        H = R_rx^{1/2} @ H_iid @ R_tx^{1/2},

    where ``H_iid`` is i.i.d. CN(0, 1) and ``R_rx``, ``R_tx`` are positive
    semi-definite Hermitian correlation matrices on the receive (rows) and
    transmit (columns) sides respectively. The Cholesky factors are cached
    once at construction.

    Args:
        shape: ``(d_out, d_in)``; must equal ``(R_rx.shape[0], R_tx.shape[0])``.
        R_rx: ``(d_out, d_out)`` Hermitian PSD receive correlation matrix.
        R_tx: ``(d_in, d_in)`` Hermitian PSD transmit correlation matrix.
        dtype: complex dtype.
        device: torch device.

    Returns:
        Callable ``sampler(B) -> (B, d_out, d_in)`` tensor.
    """
    d_out, d_in = _validate_shape(shape)
    if R_rx.shape != (d_out, d_out):
        raise ValueError(
            f"R_rx shape {tuple(R_rx.shape)} does not match (d_out, d_out) = "
            f"({d_out}, {d_out})"
        )
    if R_tx.shape != (d_in, d_in):
        raise ValueError(
            f"R_tx shape {tuple(R_tx.shape)} does not match (d_in, d_in) = "
            f"({d_in}, {d_in})"
        )
    R_rx_cast = R_rx.to(dtype=dtype, device=device)
    R_tx_cast = R_tx.to(dtype=dtype, device=device)
    # Hermitianize to mitigate input drift, then Cholesky.
    R_rx_sym = 0.5 * (R_rx_cast + R_rx_cast.mH)
    R_tx_sym = 0.5 * (R_tx_cast + R_tx_cast.mH)
    L_rx = torch.linalg.cholesky(R_rx_sym)
    L_tx = torch.linalg.cholesky(R_tx_sym)

    def sampler(batch_size: int) -> torch.Tensor:
        h_iid = _randn_complex(batch_size, d_out, d_in, dtype=dtype, device=device)
        # (d_out, d_out) @ (B, d_out, d_in) @ (d_in, d_in)^H using broadcasting.
        return L_rx @ h_iid @ L_tx.mH

    return sampler


def scaled_rayleigh(
    shape: tuple[int, int],
    sigma: float | torch.Tensor,
    *,
    dtype: torch.dtype = torch.complex128,
    device: torch.device | str | None = None,
) -> Sampler:
    """Return a scaled (large-scale fading) Rayleigh sampler.

    Each entry is CN(0, sigma^2). When ``sigma`` is a tensor, it must be
    broadcast-compatible with ``(d_out, d_in)`` to enable per-entry scaling
    (e.g., path-loss profiles).

    Args:
        shape: ``(d_out, d_in)``.
        sigma: scalar or tensor that broadcasts to ``(d_out, d_in)``.
            Real-valued and non-negative.
        dtype: complex dtype.
        device: torch device.

    Returns:
        Callable ``sampler(B) -> (B, d_out, d_in)`` tensor.
    """
    d_out, d_in = _validate_shape(shape)
    if isinstance(sigma, torch.Tensor):
        if sigma.is_complex():
            raise ValueError("sigma must be real-valued.")
        if (sigma < 0).any():
            raise ValueError("sigma entries must be non-negative.")
        sigma_real = sigma.to(
            dtype=torch.float64 if dtype == torch.complex128 else torch.float32,
            device=device,
        )
    else:
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative; got {sigma}")
        sigma_real = float(sigma)

    base_sampler = rayleigh(shape, dtype=dtype, device=device)

    def sampler(batch_size: int) -> torch.Tensor:
        h = base_sampler(batch_size)
        if isinstance(sigma_real, float):
            return sigma_real * h
        # Broadcast tensor sigma over the batch axis.
        return sigma_real.unsqueeze(0) * h

    return sampler


def constant(H_fixed: torch.Tensor) -> Sampler:
    """Return a non-fading sampler that emits a fixed channel for all draws.

    Each call broadcasts ``H_fixed`` over the batch axis, so the sampler
    is consistent with the fading-DAG API even when no fading is desired.
    The returned tensor shares storage with ``H_fixed`` via ``expand``;
    the caller must not modify it in place if the original is to be reused.

    Args:
        H_fixed: ``(d_out, d_in)`` complex tensor (any complex dtype/device).

    Returns:
        Callable ``sampler(B) -> (B, d_out, d_in)`` tensor.
    """
    if H_fixed.dim() != 2:
        raise ValueError(
            f"H_fixed must be a 2-D matrix; got shape {tuple(H_fixed.shape)}"
        )
    if not H_fixed.is_complex():
        raise ValueError(
            f"H_fixed must be a complex tensor; got dtype {H_fixed.dtype}"
        )

    def sampler(batch_size: int) -> torch.Tensor:
        return H_fixed.unsqueeze(0).expand(batch_size, *H_fixed.shape)

    return sampler
