"""Phase 4 descriptive statistics, feature tests, and binary metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def describe(values: object) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("description input must be a nonempty finite vector")
    q25, median, q75 = np.percentile(array, [25.0, 50.0, 75.0])
    return {
        "n": int(array.size),
        "mean": float(np.mean(array, dtype=np.float64)),
        "std": float(np.std(array, ddof=1, dtype=np.float64)) if array.size > 1 else 0.0,
        "median": float(median),
        "iqr": float(q75 - q25),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def cliffs_delta_e_minus_d(values_d: object, values_e: object) -> float:
    """Return Cliff's delta P(E>D)-P(E<D); ties contribute zero."""
    d = np.asarray(values_d, dtype=np.float64)
    e = np.asarray(values_e, dtype=np.float64)
    if d.ndim != 1 or e.ndim != 1 or not d.size or not e.size:
        raise ValueError("both class vectors must be nonempty and one-dimensional")
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(e)):
        raise ValueError("class vectors must be finite")
    # At Bonn scale, 100x100 pairwise comparisons are small and transparent.
    differences = e[:, None] - d[None, :]
    return float((np.count_nonzero(differences > 0) - np.count_nonzero(differences < 0)) / differences.size)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down family-wise adjusted p-values, returned in input order."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be a finite vector in [0, 1]")
    m = values.size
    order = np.argsort(values, kind="stable")
    sorted_p = values[order]
    adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * sorted_p)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(values)
    adjusted[order] = adjusted_sorted
    return adjusted


def feature_test_rows(
    feature_values: Mapping[str, Mapping[str, object]],
    feature_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Describe D/E and run two-sided Mann–Whitney tests with Holm correction."""
    interim: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for feature in feature_names:
        values_d = np.asarray(feature_values[feature]["D"], dtype=np.float64)
        values_e = np.asarray(feature_values[feature]["E"], dtype=np.float64)
        if not values_d.size or not values_e.size:
            raise ValueError(f"feature {feature} has an empty D or E group")
        test = mannwhitneyu(values_e, values_d, alternative="two-sided", method="auto")
        row: dict[str, Any] = {"feature": feature}
        for label, values in (("D", values_d), ("E", values_e)):
            for key, value in describe(values).items():
                row[f"{key}_{label}"] = value
        row["test"] = "Mann-Whitney U, two-sided"
        row["U_E"] = float(test.statistic)
        row["p_value_raw"] = float(test.pvalue)
        row["effect_size_name"] = "Cliff's delta (E minus D)"
        row["effect_size"] = cliffs_delta_e_minus_d(values_d, values_e)
        raw_p.append(float(test.pvalue))
        interim.append(row)
    adjusted = holm_adjust(raw_p)
    for row, p_adjusted in zip(interim, adjusted):
        row["p_value_holm_adjusted"] = float(p_adjusted)
    return interim


def binary_fold_metrics(y_true: object, y_pred: object, probability_e: object) -> dict[str, float]:
    """Calculate binary metrics with label 1 (E/ictal) as the positive class."""
    truth = np.asarray(y_true, dtype=np.int8)
    prediction = np.asarray(y_pred, dtype=np.int8)
    probability = np.asarray(probability_e, dtype=np.float64)
    if truth.ndim != 1 or prediction.shape != truth.shape or probability.shape != truth.shape:
        raise ValueError("truth, prediction, and probability vectors must have matching 1D shapes")
    if not np.all(np.isfinite(probability)):
        raise ValueError("probability_e contains NaN or Inf")
    true_e = truth == 1
    pred_e = prediction == 1
    tp = int(np.count_nonzero(true_e & pred_e))
    fn = int(np.count_nonzero(true_e & ~pred_e))
    tn = int(np.count_nonzero(~true_e & ~pred_e))
    fp = int(np.count_nonzero(~true_e & pred_e))
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "F1": float(f1_score(truth, prediction, pos_label=1, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(truth, probability)),
    }
