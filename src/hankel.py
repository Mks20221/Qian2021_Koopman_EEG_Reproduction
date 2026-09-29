"""Hankel delay embedding and an explicit truncated SVD."""

from __future__ import annotations

import numpy as np


def _real_finite_array(values: object, name: str, ndim: int) -> np.ndarray:
    """Return a finite float64 array, rejecting complex and nonnumeric input."""
    array = np.asarray(values)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional; got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError(f"{name} must contain real numeric values")
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must not contain NaN or Inf")
    return array


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return int(value)


def build_hankel(x: object, delay_dim: int) -> np.ndarray:
    """Build H[i, j] = x[i + j] with shape [delay_dim, T-delay_dim+1].

    Rows are filled explicitly so the delay convention is visible and does not
    depend on a Hankel helper with a different row/column convention.
    """
    signal = _real_finite_array(x, "x", ndim=1)
    delay_dim = _positive_integer(delay_dim, "delay_dim")
    if delay_dim > signal.size:
        raise ValueError(
            f"delay_dim ({delay_dim}) cannot exceed signal length ({signal.size})"
        )

    n_columns = signal.size - delay_dim + 1
    matrix = np.empty((delay_dim, n_columns), dtype=np.float64)
    for row in range(delay_dim):
        matrix[row, :] = signal[row : row + n_columns]
    return matrix


def truncated_svd(
    H: object, rank: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (U_r, S_r, Vh_r, singular_values_full).

    The untruncated singular-value vector is returned as the fourth value for
    diagnostics. This function does not select an intrinsic-coordinate
    convention; that choice belongs to ``build_intrinsic_coordinates``.
    """
    matrix = _real_finite_array(H, "H", ndim=2)
    if min(matrix.shape, default=0) < 1:
        raise ValueError("H must have nonzero dimensions")
    rank = _positive_integer(rank, "rank")
    max_rank = min(matrix.shape)
    if rank > max_rank:
        raise ValueError(f"rank ({rank}) cannot exceed min(H.shape) ({max_rank})")

    try:
        U, singular_values, Vh = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(f"SVD failed for H with shape {matrix.shape}") from exc

    if not (
        np.all(np.isfinite(U))
        and np.all(np.isfinite(singular_values))
        and np.all(np.isfinite(Vh))
    ):
        raise FloatingPointError("SVD returned a non-finite value")

    return (
        U[:, :rank].copy(),
        singular_values[:rank].copy(),
        Vh[:rank, :].copy(),
        singular_values.copy(),
    )
