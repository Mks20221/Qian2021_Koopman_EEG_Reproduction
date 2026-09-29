"""Finite-dimensional least-squares Koopman/DMD diagnostics."""

from __future__ import annotations

import numpy as np


def _real_finite_array(values: object, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional; got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError(f"{name} must contain real numeric values")
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must not contain NaN or Inf")
    return array


def _finite_eigenvalues(values: object) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"eigvals must be one-dimensional; got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("eigvals must be numeric")
    values_complex = np.asarray(array, dtype=np.complex128)
    if not np.all(np.isfinite(values_complex)):
        raise ValueError("eigvals must not contain NaN or Inf")
    return values_complex


def build_intrinsic_coordinates(
    U_r: object,
    S_r: object,
    Vh_r: object,
    source: str = "V",
) -> np.ndarray:
    """Build time-indexed intrinsic-coordinate candidates.

    ``V = Vh_r.T`` has shape [N, r], so its time rows are returned as
    ``V.T`` with shape [r, N]. ``SigmaV`` returns ``diag(S_r) @ Vh_r``.
    Neither candidate is asserted to be the paper's uniquely specified y.
    ``U_r`` is validated for consistency but is not a time-indexed candidate.
    """
    U = _real_finite_array(U_r, "U_r", ndim=2)
    singular_values = _real_finite_array(S_r, "S_r", ndim=1)
    Vh = _real_finite_array(Vh_r, "Vh_r", ndim=2)
    r = singular_values.size
    if r < 1:
        raise ValueError("S_r must contain at least one singular value")
    if U.shape[1] != r or Vh.shape[0] != r:
        raise ValueError(
            "inconsistent truncated SVD shapes: expected U_r.shape[1] == "
            "len(S_r) == Vh_r.shape[0]"
        )
    if U.shape[0] < r or Vh.shape[1] < 1:
        raise ValueError("truncated SVD arrays must have nonzero sample dimensions")

    if source == "V":
        V = Vh.T
        coordinates = V.T
    elif source == "SigmaV":
        try:
            with np.errstate(over="raise", invalid="raise"):
                coordinates = singular_values[:, None] * Vh
        except FloatingPointError as exc:
            raise FloatingPointError("SigmaV coordinate construction overflowed") from exc
    else:
        raise ValueError("source must be 'V' or 'SigmaV'")

    if not np.all(np.isfinite(coordinates)):
        raise FloatingPointError("intrinsic coordinates contain a non-finite value")
    return np.asarray(coordinates, dtype=np.float64)


def fit_linear_dynamics(Y: object) -> dict[str, object]:
    """Fit the unregularized snapshot map A = Y2 @ pinv(Y1).

    No eigenvalue sorting, clipping, stabilization, TLS, or forward-backward
    correction is applied. ``Y`` is real-valued; eigenvalues may be complex.
    """
    coordinates = _real_finite_array(Y, "Y", ndim=2)
    if coordinates.shape[0] < 1 or coordinates.shape[1] < 2:
        raise ValueError("Y must have shape [r, N] with r >= 1 and N >= 2")

    Y1 = coordinates[:, :-1]
    Y2 = coordinates[:, 1:]
    try:
        with np.errstate(over="raise", invalid="raise"):
            A = Y2 @ np.linalg.pinv(Y1)
            Y2_hat = A @ Y1
        eigvals, eigvecs = np.linalg.eig(A)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError("least-squares DMD or eigendecomposition failed") from exc
    except FloatingPointError as exc:
        raise FloatingPointError("least-squares DMD produced a numerical overflow") from exc

    if not (
        np.all(np.isfinite(A))
        and np.all(np.isfinite(Y2_hat))
        and np.all(np.isfinite(eigvals))
        and np.all(np.isfinite(eigvecs))
    ):
        raise FloatingPointError("linear dynamics fit returned a non-finite value")

    residual_norm = float(np.linalg.norm(Y2 - Y2_hat, ord="fro"))
    target_norm = float(np.linalg.norm(Y2, ord="fro"))
    if not (np.isfinite(residual_norm) and np.isfinite(target_norm)):
        raise FloatingPointError("linearity-error norm is not finite")
    linearity_error = 0.0 if target_norm == 0.0 else residual_norm / target_norm
    if not np.isfinite(linearity_error):
        raise FloatingPointError("linearity error is not finite")

    return {
        "A": np.asarray(A, dtype=np.float64),
        "eigvals": np.asarray(eigvals, dtype=np.complex128),
        "eigvecs": np.asarray(eigvecs, dtype=np.complex128),
        "linearity_error": float(linearity_error),
        "Y1": Y1.copy(),
        "Y2": Y2.copy(),
        "Y2_hat": np.asarray(Y2_hat, dtype=np.float64),
    }


def eigenvalue_diagnostics(eigvals: object, dt: float) -> dict[str, np.ndarray]:
    """Return discrete and continuous-time eigenvalue diagnostics.

    For zero eigenvalues, ``mu``, frequency, and growth rate are NaN and the
    corresponding ``continuous_time_valid`` entry is False. The eigenvalue is
    retained in the output and is not clipped or removed.
    """
    values = _finite_eigenvalues(eigvals)
    if isinstance(dt, (bool, np.bool_)) or not np.isscalar(dt):
        raise TypeError("dt must be a positive finite real scalar")
    if np.iscomplexobj(dt):
        raise TypeError("dt must be real")
    dt_value = float(dt)
    if not np.isfinite(dt_value) or dt_value <= 0.0:
        raise ValueError("dt must be positive and finite")

    magnitude = np.abs(values).astype(np.float64)
    phase = np.angle(values).astype(np.float64)
    if not (np.all(np.isfinite(magnitude)) and np.all(np.isfinite(phase))):
        raise FloatingPointError("eigenvalue magnitude or phase is not finite")
    valid = values != 0.0
    mu = np.full(values.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    if np.any(valid):
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                mu[valid] = np.log(values[valid]) / dt_value
        except FloatingPointError as exc:
            raise FloatingPointError("continuous-time eigenvalue conversion overflowed") from exc
        if not np.all(np.isfinite(mu[valid])):
            raise FloatingPointError("continuous-time eigenvalue conversion is not finite")
    frequency_hz = np.imag(mu) / (2.0 * np.pi)
    growth_rate = np.real(mu)
    if not (
        np.all(np.isfinite(frequency_hz[valid]))
        and np.all(np.isfinite(growth_rate[valid]))
    ):
        raise FloatingPointError("frequency or growth-rate diagnostic is not finite")
    return {
        "magnitude": magnitude,
        "phase": phase,
        "mu": mu,
        "frequency_hz": np.asarray(frequency_hz, dtype=np.float64),
        "growth_rate": np.asarray(growth_rate, dtype=np.float64),
        "continuous_time_valid": valid,
    }


def sort_eigenvalues_by_magnitude(eigvals: object) -> np.ndarray:
    """Return descending-|lambda| indices for diagnostics only.

    This index order never changes or filters an intrinsic-coordinate matrix.
    """
    values = _finite_eigenvalues(eigvals)
    return np.argsort(np.abs(values))[::-1]
