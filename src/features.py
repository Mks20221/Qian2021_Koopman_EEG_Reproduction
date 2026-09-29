"""Small, explicit summary-feature calculations for Phase 3 smoke diagnostics."""

from __future__ import annotations

import numpy as np


def shannon_histogram_entropy(values: object, bins: int = 64) -> float:
    """Compute natural-log Shannon entropy from a fixed-count per-signal histogram.

    Histogram range follows the current signal's observed range. This is an
    implementation choice, not a paper-defined estimator.
    """
    array = _finite_vector(values, "values")
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("bins must be a positive integer")
    counts, _ = np.histogram(array, bins=int(bins))
    nonzero = counts[counts > 0].astype(np.float64)
    probabilities = nonzero / float(array.size)
    return float(-np.sum(probabilities * np.log(probabilities), dtype=np.float64))


def _finite_vector(values: object, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{name} must be a real numeric one-dimensional array")
    array = np.asarray(array, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be nonempty and finite")
    return array


def summary_features(values: object, bins: int = 64) -> dict[str, float]:
    """Return max/min/mean/population std/mean-square power/entropy."""
    array = _finite_vector(values, "values")
    with np.errstate(over="raise", invalid="raise"):
        power = float(np.mean(np.square(array), dtype=np.float64))
    features = {
        "max": float(np.max(array)),
        "min": float(np.min(array)),
        "mean": float(np.mean(array, dtype=np.float64)),
        "std": float(np.std(array, ddof=0, dtype=np.float64)),
        "power": power,
        "entropy": shannon_histogram_entropy(array, bins=bins),
    }
    if not all(np.isfinite(value) for value in features.values()):
        raise FloatingPointError("summary feature calculation produced NaN or Inf")
    return features


def raw_summary_features(values: object, bins: int = 64) -> dict[str, float]:
    summary = summary_features(values, bins=bins)
    return {
        "raw_mean": summary["mean"],
        "raw_std": summary["std"],
        "raw_power": summary["power"],
        "raw_entropy": summary["entropy"],
    }
