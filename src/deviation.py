"""Explicit candidate distances for intrinsic-space dynamical deviation.

Using absolute eigenvalue weights is an implementation choice for converting
the paper's eigenvalue-weighted Eq. (8) expression to real nonnegative weights;
it is not confirmed as the authors' numerical handling.
"""

from __future__ import annotations

import numpy as np


def _real_finite_vector(values: object, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError(f"{name} must contain real numeric values")
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must not contain NaN or Inf")
    return array


def _difference(x: object, reference: object) -> np.ndarray:
    values = _real_finite_vector(x, "x")
    ref = _real_finite_vector(reference, "reference")
    if values.shape != ref.shape:
        raise ValueError("x and reference must have the same shape")
    try:
        with np.errstate(over="raise", invalid="raise"):
            difference = np.abs(values - ref)
    except FloatingPointError as exc:
        raise FloatingPointError("x-reference overflowed") from exc
    if not np.all(np.isfinite(difference)):
        raise FloatingPointError("distance difference contains a non-finite value")
    return difference


def _validate_weights(weights: object, n_dimensions: int) -> np.ndarray:
    raw = np.asarray(weights)
    if raw.ndim != 1 or raw.size != n_dimensions:
        raise ValueError(f"weights must have shape [{n_dimensions}]")
    if np.iscomplexobj(raw):
        raise TypeError("weights must be real; complex weights are not valid distances")
    if not np.issubdtype(raw.dtype, np.number):
        raise TypeError("weights must be real numeric values")
    real_weights = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(real_weights)):
        raise ValueError("weights must be finite")
    if np.any(real_weights < 0.0):
        raise ValueError("weights must be nonnegative")
    return real_weights


def _validate_p(p: object) -> float:
    if isinstance(p, (bool, np.bool_)) or not np.isscalar(p) or np.iscomplexobj(p):
        raise TypeError("p must be a positive finite real scalar")
    try:
        p_value = float(p)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("p must be a positive finite real scalar") from exc
    if not np.isfinite(p_value) or p_value <= 0.0:
        raise ValueError("p must be positive and finite")
    return p_value


def finite_p_weighted_distance(
    x: object, reference: object, weights: object, p: float
) -> float:
    r"""Compute ``(sum_i w_i |x_i-reference_i|**p)**(1/p)`` stably.

    A log-sum-exp calculation is used after scaling differences by their
    maximum magnitude. This avoids directly raising large raw differences to
    a high power. The result is float64; an unrepresentable result raises.
    """
    difference = _difference(x, reference)
    real_weights = _validate_weights(weights, difference.size)
    p_value = _validate_p(p)
    maximum_difference = float(np.max(difference))
    if maximum_difference == 0.0 or not np.any(real_weights > 0.0):
        return 0.0

    active = (difference > 0.0) & (real_weights > 0.0)
    log_ratio = np.log(difference[active]) - np.log(maximum_difference)
    log_weight = np.log(real_weights[active])
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        log_terms = log_weight + p_value * log_ratio

    max_log_term = float(np.max(log_terms))
    if np.isneginf(max_log_term):
        return 0.0
    scaled_sum = float(np.sum(np.exp(log_terms - max_log_term), dtype=np.float64))
    if not np.isfinite(scaled_sum) or scaled_sum <= 0.0:
        raise FloatingPointError("weighted p-sum could not be represented")

    log_distance = np.log(maximum_difference) + (
        max_log_term + np.log(scaled_sum)
    ) / p_value
    max_log_float = np.log(np.finfo(np.float64).max)
    min_log_subnormal = np.log(np.nextafter(np.float64(0.0), np.float64(1.0)))
    if log_distance > max_log_float:
        raise FloatingPointError("weighted p-distance exceeds float64 range")
    if log_distance < min_log_subnormal:
        return 0.0
    distance = float(np.exp(log_distance))
    if not np.isfinite(distance):
        raise FloatingPointError("weighted p-distance is not finite")
    return distance


def chebyshev_distance(x: object, reference: object) -> float:
    """Compute the unweighted infinity norm ``max_i |x_i-reference_i|``."""
    difference = _difference(x, reference)
    return float(np.max(difference))


def make_eigenvalue_weights(eigvals: object, mode: str = "abs") -> np.ndarray:
    """Return real nonnegative eigenvalue weights for an explicit mode.

    ``mode='abs'`` is an implementation choice for Eq. (8), not a verified
    statement of the paper's numerical handling. No other mode is supported.
    """
    values = np.asarray(eigvals)
    if values.ndim != 1:
        raise ValueError("eigvals must be one-dimensional")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("eigvals must be numeric")
    values = np.asarray(values, dtype=np.complex128)
    if not np.all(np.isfinite(values)):
        raise ValueError("eigvals must not contain NaN or Inf")
    if mode != "abs":
        raise ValueError("only mode='abs' is supported in Phase 1")
    weights = np.abs(values).astype(np.float64)
    if not np.all(np.isfinite(weights)):
        raise FloatingPointError("eigenvalue magnitudes are not finite")
    return weights


def trajectory_deviation(
    Y: object,
    reference: object | None = None,
    distance_mode: str = "chebyshev",
    weights: object | None = None,
    p: float = 20.0,
    reference_mode: str = "origin",
) -> np.ndarray:
    """Compute a distance at each time from an explicit reference vector.

    Supported reference modes are ``origin``, ``trajectory_centroid``, and
    ``explicit``. ``training_reference_centroid`` is deliberately not
    implemented because it requires a training-fold data flow.
    """
    coordinates = np.asarray(Y)
    if coordinates.ndim != 2:
        raise ValueError(f"Y must have shape [r, N]; got shape {coordinates.shape}")
    if not np.issubdtype(coordinates.dtype, np.number) or np.iscomplexobj(coordinates):
        raise TypeError("Y must contain real numeric values")
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Y must not contain NaN or Inf")
    n_dimensions, n_times = coordinates.shape
    if n_dimensions < 1 or n_times < 1:
        raise ValueError("Y must have at least one coordinate and one time point")

    if reference_mode == "origin":
        if reference is not None:
            raise ValueError("do not pass reference with reference_mode='origin'")
        reference_vector = np.zeros(n_dimensions, dtype=np.float64)
    elif reference_mode == "trajectory_centroid":
        if reference is not None:
            raise ValueError(
                "do not pass reference with reference_mode='trajectory_centroid'"
            )
        reference_vector = np.mean(coordinates, axis=1, dtype=np.float64)
        if not np.all(np.isfinite(reference_vector)):
            raise FloatingPointError("trajectory centroid is not finite")
    elif reference_mode == "explicit":
        if reference is None:
            raise ValueError("reference is required for reference_mode='explicit'")
        reference_vector = _real_finite_vector(reference, "reference")
        if reference_vector.size != n_dimensions:
            raise ValueError("reference length must match Y coordinate dimension")
    elif reference_mode == "training_reference_centroid":
        raise NotImplementedError(
            "training_reference_centroid requires a training-fold data flow and is not "
            "implemented in Phase 1"
        )
    else:
        raise ValueError(
            "reference_mode must be 'origin', 'trajectory_centroid', or 'explicit'"
        )

    if distance_mode == "chebyshev":
        distance_function = lambda point: chebyshev_distance(point, reference_vector)
    elif distance_mode == "paper_literal_finite_p":
        if weights is None:
            raise ValueError("weights are required for paper_literal_finite_p")
        real_weights = _validate_weights(weights, n_dimensions)
        p_value = _validate_p(p)
        distance_function = lambda point: finite_p_weighted_distance(
            point, reference_vector, real_weights, p_value
        )
    else:
        raise ValueError(
            "distance_mode must be 'chebyshev' or 'paper_literal_finite_p'"
        )

    deviations = np.empty(n_times, dtype=np.float64)
    for time_index in range(n_times):
        deviations[time_index] = distance_function(coordinates[:, time_index])
    if not np.all(np.isfinite(deviations)):
        raise FloatingPointError("trajectory deviation returned a non-finite value")
    return deviations
