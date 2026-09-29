"""Phase 4: fixed full Bonn D/E feature extraction and segment-level evaluation."""

from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
import imblearn
from imblearn.ensemble import RUSBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import discover_bonn_files, load_bonn_segment
from src.deviation import make_eigenvalue_weights, trajectory_deviation
from src.evaluate import binary_fold_metrics, describe, feature_test_rows
from src.features import raw_summary_features, summary_features
from src.hankel import build_hankel, truncated_svd
from src.koopman import build_intrinsic_coordinates, eigenvalue_diagnostics, fit_linear_dynamics


KOOPMAN_FEATURES = ["Dmax_cheb", "Dmin_cheb", "Dmean_cheb", "Dstd_cheb", "Dpower_cheb", "Dentropy_cheb"]
RAW_FEATURES = ["raw_mean", "raw_std", "raw_power", "raw_entropy"]
FINITE_P_FEATURES = ["Dmax_fp", "Dmin_fp", "Dmean_fp", "Dstd_fp", "Dpower_fp", "Dentropy_fp"]
METRIC_NAMES = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "F1", "ROC_AUC"]


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"cannot encode {type(value).__name__} as JSON")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]], columns: list[str] | None = None) -> None:
    if not rows and not columns:
        raise ValueError(f"CSV {path} needs explicit columns when it has no rows")
    fieldnames = columns or list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _process_signal(signal: np.ndarray, config: dict[str, Any], fs: float) -> dict[str, Any]:
    """Run the exact fixed Phase 3 Hankel/SVD/V/A/eigenvalue/D/features sequence."""
    result: dict[str, Any] = {
        "segment_length": int(signal.size),
        "H_shape": "",
        "singular_values": "",
        "linearity_error": None,
        "spectral_radius": None,
        "nan_inf_occurred": not bool(np.all(np.isfinite(signal))),
        "svd_attempted": False,
        "svd_succeeded": False,
        "eig_attempted": False,
        "eig_succeeded": False,
        "pipeline_success": False,
        "error_reason": "",
        "features": {},
        "D_cheb": None,
        "D_finite_p": None,
        "Y": None,
    }
    try:
        if result["nan_inf_occurred"]:
            raise ValueError("input signal contains NaN or Inf")
        result["features"].update(raw_summary_features(signal, bins=int(config["entropy_bins"])))
        H = build_hankel(signal, int(config["delay_dim"]))
        result["H_shape"] = f"{H.shape[0]}x{H.shape[1]}"
        result["svd_attempted"] = True
        U_r, S_r, Vh_r, singular_values = truncated_svd(H, int(config["svd_rank"]))
        result["svd_succeeded"] = True
        result["singular_values"] = json.dumps(singular_values.tolist(), separators=(",", ":"))
        Y = build_intrinsic_coordinates(U_r, S_r, Vh_r, source=str(config["coordinate_source"]))
        if Y.shape[0] != int(config["koopman_coordinate_count"]):
            raise RuntimeError(f"coordinate count mismatch: got {Y.shape[0]}")
        result["Y"] = Y
        result["eig_attempted"] = True
        fit = fit_linear_dynamics(Y)
        result["linearity_error"] = float(fit["linearity_error"])
        result["eig_succeeded"] = True
        eigen = eigenvalue_diagnostics(fit["eigvals"], dt=1.0 / fs)
        result["spectral_radius"] = float(np.max(eigen["magnitude"]))
        weights = make_eigenvalue_weights(fit["eigvals"], mode=str(config["finite_p_weight_mode"]))
        D_cheb = trajectory_deviation(Y, reference_mode=str(config["reference_mode"]), distance_mode="chebyshev")
        D_finite_p = trajectory_deviation(
            Y,
            reference_mode=str(config["reference_mode"]),
            distance_mode="paper_literal_finite_p",
            weights=weights,
            p=float(config["finite_p"]),
        )
        if not all(np.all(np.isfinite(a)) for a in (H, Y, fit["A"], fit["eigvals"], D_cheb, D_finite_p)):
            result["nan_inf_occurred"] = True
            raise FloatingPointError("NaN or Inf found in an intermediate result")
        result["D_cheb"] = D_cheb
        result["D_finite_p"] = D_finite_p
        cheb = summary_features(D_cheb, bins=int(config["entropy_bins"]))
        finite_p = summary_features(D_finite_p, bins=int(config["entropy_bins"]))
        result["features"].update({
            "Dmax_cheb": cheb["max"], "Dmin_cheb": cheb["min"],
            "Dmean_cheb": cheb["mean"], "Dstd_cheb": cheb["std"],
            "Dpower_cheb": cheb["power"], "Dentropy_cheb": cheb["entropy"],
            "Dmax_fp": finite_p["max"], "Dmin_fp": finite_p["min"],
            "Dmean_fp": finite_p["mean"], "Dstd_fp": finite_p["std"],
            "Dpower_fp": finite_p["power"], "Dentropy_fp": finite_p["entropy"],
        })
        result["pipeline_success"] = True
    except Exception as exc:
        result["error_reason"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, FloatingPointError) or "NaN" in str(exc) or "Inf" in str(exc):
            result["nan_inf_occurred"] = True
    return result


def _metric_row(sample_id: str, label: str, state: str, path: Path, result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "label": label,
        "state": state,
        "file_path": str(path),
        "segment_length": result["segment_length"],
        "delay_dim": config["delay_dim"],
        "svd_rank": config["svd_rank"],
        "coordinate_count": config["koopman_coordinate_count"],
        "coordinate_source": config["coordinate_source"],
        "reference_mode": config["reference_mode"],
        "H_shape": result["H_shape"],
        "singular_values": result["singular_values"],
        "linearity_error": result["linearity_error"],
        "spectral_radius": result["spectral_radius"],
        "nan_inf_occurred": result["nan_inf_occurred"],
        "svd_attempted": result["svd_attempted"],
        "svd_succeeded": result["svd_succeeded"],
        "eig_attempted": result["eig_attempted"],
        "eig_succeeded": result["eig_succeeded"],
        "pipeline_success": result["pipeline_success"],
        "error_reason": result["error_reason"],
    }


def _scale_control(sample_signals: dict[str, np.ndarray], original_results: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    factor = float(config["scale_invariance_control"]["scale_factor"])
    report: dict[str, Any] = {
        "control": "x(t) vs factor*x(t), same fixed Phase 3 pipeline, V coordinates, Chebyshev D",
        "scale_factor": factor,
        "sample_ids": list(config["scale_invariance_control"]["sample_ids"]),
        "samples": {},
    }
    epsilon = np.finfo(np.float64).eps
    for sample_id in report["sample_ids"]:
        original_signal = sample_signals[sample_id]
        original = original_results[sample_id]
        scaled = _process_signal(original_signal * factor, config, float(config["sampling_rate_hz"]))
        if not original["pipeline_success"] or not scaled["pipeline_success"]:
            report["samples"][sample_id] = {
                "success": False,
                "original_error": original["error_reason"],
                "scaled_error": scaled["error_reason"],
            }
            continue
        feature_names = list(config["scale_invariance_control"]["features"])
        original_features = original["features"]
        scaled_features = scaled["features"]
        relative: dict[str, float] = {}
        for feature in feature_names:
            base = float(original_features[feature])
            changed = float(scaled_features[feature])
            relative[feature] = abs(changed - base) / max(abs(base), epsilon)
        raw_relative = {
            feature: abs(float(scaled_features[feature]) - float(original_features[feature]))
            / max(abs(float(original_features[feature])), epsilon)
            for feature in RAW_FEATURES
        }
        report["samples"][sample_id] = {
            "success": True,
            "original_H_shape": original["H_shape"],
            "scaled_H_shape": scaled["H_shape"],
            "koopman_features_original": {feature: original_features[feature] for feature in feature_names},
            "koopman_features_scaled": {feature: scaled_features[feature] for feature in feature_names},
            "koopman_feature_relative_difference": relative,
            "max_koopman_feature_relative_difference": max(relative.values()),
            "raw_feature_relative_difference": raw_relative,
            "original_linearity_error": original["linearity_error"],
            "scaled_linearity_error": scaled["linearity_error"],
            "original_spectral_radius": original["spectral_radius"],
            "scaled_spectral_radius": scaled["spectral_radius"],
        }
    return report


def _group_feature_values(rows: list[dict[str, Any]], features: list[str]) -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for feature in features:
        grouped[feature] = {"D": [], "E": []}
        for row in rows:
            value = row.get(feature)
            if value is not None and value != "":
                grouped[feature][str(row["label"])].append(float(value))
    return grouped


def _linearity_summary(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in ("linearity_error", "spectral_radius"):
        for label in ("D", "E"):
            values = [float(row[metric]) for row in metrics if row["label"] == label and row[metric] not in (None, "")]
            stats = describe(values)
            rows.append({"metric": metric, "label": label, **stats})
    return rows


def _draw_distribution(ax: plt.Axes, groups: list[np.ndarray], ylabel: str, title: str) -> None:
    ax.boxplot(groups, labels=["D / interictal", "E / ictal"], showfliers=False, widths=0.5)
    colors = ("#2563eb", "#dc2626")
    for xpos, (values, color) in enumerate(zip(groups, colors), start=1):
        jitter = np.linspace(-0.12, 0.12, values.size) if values.size > 1 else np.array([0.0])
        ax.scatter(xpos + jitter, values, s=10, alpha=0.55, color=color, edgecolors="none")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.2)


def _plot_koopman_distributions(feature_rows: list[dict[str, Any]], out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feature in zip(axes.flat, KOOPMAN_FEATURES):
        d = np.asarray([float(row[feature]) for row in feature_rows if row["label"] == "D"], dtype=float)
        e = np.asarray([float(row[feature]) for row in feature_rows if row["label"] == "E"], dtype=float)
        _draw_distribution(ax, [d, e], feature, feature)
    fig.suptitle("Bonn full: primary Chebyshev Koopman feature distributions")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_dmax_dstd(feature_rows: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for label, color, name in (("D", "#2563eb", "D / interictal"), ("E", "#dc2626", "E / ictal")):
        selected = [row for row in feature_rows if row["label"] == label]
        ax.scatter([float(row["Dmax_cheb"]) for row in selected], [float(row["Dstd_cheb"]) for row in selected], s=24, alpha=0.72, color=color, label=name)
    ax.set(xlabel="Dmax (Chebyshev)", ylabel="Dstd (Chebyshev)", title="Dmax vs Dstd on all 200 segments")
    ax.grid(alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_raw_distributions(feature_rows: list[dict[str, Any]], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, feature in zip(axes.flat, RAW_FEATURES):
        d = np.asarray([float(row[feature]) for row in feature_rows if row["label"] == "D"], dtype=float)
        e = np.asarray([float(row[feature]) for row in feature_rows if row["label"] == "E"], dtype=float)
        _draw_distribution(ax, [d, e], feature, feature)
    fig.suptitle("Bonn full: raw EEG statistical baseline (amplitude/power differences matter)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_linearity(metrics: list[dict[str, Any]], out: Path) -> None:
    d = np.asarray([float(row["linearity_error"]) for row in metrics if row["label"] == "D" and row["linearity_error"] not in (None, "")])
    e = np.asarray([float(row["linearity_error"]) for row in metrics if row["label"] == "E" and row["linearity_error"] not in (None, "")])
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    _draw_distribution(ax, [d, e], "Relative one-step linearity error", "Intrinsic-space linearity error")
    fig.suptitle("Bonn full: segment-level linearizability by state")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_classifier_summary(summary: list[dict[str, Any]], out: Path) -> None:
    feature_sets = ["KOOPMAN_PRIMARY", "RAW_BASELINE", "KOOPMAN_PLUS_RAW"]
    models = ["LogisticRegression", "SVM", "RUSBoost"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, metric, title in ((axes[0], "balanced_accuracy", "Balanced accuracy"), (axes[1], "ROC_AUC", "ROC-AUC")):
        offsets = np.linspace(-0.18, 0.18, len(models))
        for model_index, model in enumerate(models):
            rows = {(row["feature_set"], row["model"]): row for row in summary}
            means = [float(rows[(feature_set, model)][f"{metric}_mean"]) for feature_set in feature_sets]
            stds = [float(rows[(feature_set, model)][f"{metric}_std"]) for feature_set in feature_sets]
            ax.errorbar(np.arange(3) + offsets[model_index], means, yerr=stds, marker="o", capsize=3, lw=1.2, label=model)
        ax.set_xticks(range(3), ["Koopman", "Raw", "Koopman + Raw"], rotation=12)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"{title} (CV mean ± SD)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend()
    fig.suptitle("Repeated stratified 5×5 CV; identical folds for all models and feature sets")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_classifier(model_name: str, params: dict[str, Any]) -> Any:
    if model_name == "LogisticRegression":
        estimator = LogisticRegression(
            C=float(params["C"]), solver=str(params["solver"]), max_iter=int(params["max_iter"]),
            class_weight=params["class_weight"], random_state=int(params["random_state"]),
        )
        return Pipeline([("scaler", StandardScaler()), ("classifier", estimator)])
    if model_name == "SVM":
        estimator = SVC(
            kernel=str(params["kernel"]), C=float(params["C"]), gamma=params["gamma"],
            probability=bool(params["probability"]), class_weight=params["class_weight"],
            random_state=int(params["random_state"]),
        )
        return Pipeline([("scaler", StandardScaler()), ("classifier", estimator)])
    if model_name == "RUSBoost":
        estimator = DecisionTreeClassifier(max_depth=1, random_state=int(params["random_state"]))
        return RUSBoostClassifier(
            estimator=estimator,
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            algorithm=str(params["algorithm"]),
            sampling_strategy=str(params["sampling_strategy"]),
            replacement=bool(params["replacement"]),
            random_state=int(params["random_state"]),
        )
    raise ValueError(f"unsupported fixed classifier: {model_name}")


def _cv_evaluation(
    feature_rows: list[dict[str, Any]],
    feature_sets: dict[str, list[str]],
    classifier_params: dict[str, dict[str, Any]],
    cv_config: dict[str, Any],
    out: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sample_ids = np.asarray([str(row["sample_id"]) for row in feature_rows], dtype=object)
    labels = np.asarray([0 if row["label"] == "D" else 1 for row in feature_rows], dtype=np.int8)
    splitter = RepeatedStratifiedKFold(
        n_splits=int(cv_config["n_splits"]),
        n_repeats=int(cv_config["n_repeats"]),
        random_state=int(cv_config["random_seed"]),
    )
    split_specs: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    manifest: list[dict[str, Any]] = []
    for split_index, (train_index, test_index) in enumerate(splitter.split(np.zeros(labels.size), labels)):
        repeat = split_index // int(cv_config["n_splits"]) + 1
        fold = split_index % int(cv_config["n_splits"]) + 1
        split_specs.append((repeat, fold, train_index, test_index))
        manifest.append({
            "repeat": repeat,
            "fold": fold,
            "train_n": int(train_index.size),
            "test_n": int(test_index.size),
            "train_sample_ids": json.dumps(sample_ids[train_index].tolist(), separators=(",", ":")),
            "test_sample_ids": json.dumps(sample_ids[test_index].tolist(), separators=(",", ":")),
        })
    _write_csv(out / "cv_split_manifest.csv", manifest)

    cv_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    for model_name, params in classifier_params.items():
        for feature_set_name, columns in feature_sets.items():
            X = np.asarray([[float(row[column]) for column in columns] for row in feature_rows], dtype=np.float64)
            if not np.all(np.isfinite(X)):
                raise ValueError(f"feature set {feature_set_name} contains NaN/Inf")
            for repeat, fold, train_index, test_index in split_specs:
                estimator = _build_classifier(model_name, params)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = labels[train_index], labels[test_index]
                estimator.fit(X_train, y_train)
                y_pred = np.asarray(estimator.predict(X_test), dtype=np.int8)
                probabilities = np.asarray(estimator.predict_proba(X_test), dtype=np.float64)
                classes = np.asarray(estimator.classes_, dtype=np.int8)
                positive_col = int(np.flatnonzero(classes == 1)[0])
                probability_e = probabilities[:, positive_col]
                decision_score = np.full(test_index.size, np.nan, dtype=np.float64)
                if hasattr(estimator, "decision_function"):
                    try:
                        score = np.asarray(estimator.decision_function(X_test), dtype=np.float64)
                        if score.ndim == 1:
                            decision_score = score
                    except (AttributeError, NotImplementedError, ValueError):
                        pass
                fold_metrics = binary_fold_metrics(y_test, y_pred, probability_e)
                cv_rows.append({
                    "repeat": repeat,
                    "fold": fold,
                    "model": model_name,
                    "feature_set": feature_set_name,
                    "train_n": int(train_index.size),
                    "test_n": int(test_index.size),
                    **fold_metrics,
                })
                for local_index, global_index in enumerate(test_index):
                    oof_rows.append({
                        "sample_id": sample_ids[global_index],
                        "label": "E" if labels[global_index] == 1 else "D",
                        "repeat": repeat,
                        "fold": fold,
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "predicted_label": "E" if y_pred[local_index] == 1 else "D",
                        "predicted_probability": float(probability_e[local_index]),
                        "decision_score": float(decision_score[local_index]) if np.isfinite(decision_score[local_index]) else "",
                    })
            print(f"Completed CV: {model_name} × {feature_set_name}")

    _write_csv(out / "cv_metrics.csv", cv_rows)
    _write_csv(out / "oof_predictions.csv", oof_rows)
    summary: list[dict[str, Any]] = []
    for feature_set_name in feature_sets:
        for model_name in classifier_params:
            selected = [row for row in cv_rows if row["feature_set"] == feature_set_name and row["model"] == model_name]
            summary_row: dict[str, Any] = {
                "feature_set": feature_set_name,
                "model": model_name,
                "n_folds": len(selected),
            }
            for metric in METRIC_NAMES:
                values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
                summary_row[f"{metric}_mean"] = float(np.mean(values))
                summary_row[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            summary.append(summary_row)
    _write_csv(out / "classifier_summary.csv", summary)
    return cv_rows, oof_rows, manifest, summary


def _paired_model_comparison(cv_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    model_names = sorted({str(row["model"]) for row in cv_rows})
    for model in model_names:
        for metric in ("balanced_accuracy", "ROC_AUC"):
            raw = {(int(row["repeat"]), int(row["fold"])): float(row[metric]) for row in cv_rows if row["model"] == model and row["feature_set"] == "RAW_BASELINE"}
            combined = {(int(row["repeat"]), int(row["fold"])): float(row[metric]) for row in cv_rows if row["model"] == model and row["feature_set"] == "KOOPMAN_PLUS_RAW"}
            differences = np.asarray([combined[key] - raw[key] for key in sorted(raw)], dtype=np.float64)
            comparisons.append({
                "model": model,
                "metric": metric,
                "paired_folds": int(differences.size),
                "combined_minus_raw_mean": float(np.mean(differences)),
                "combined_minus_raw_std": float(np.std(differences, ddof=1)),
                "folds_combined_higher": int(np.count_nonzero(differences > 0)),
                "folds_equal": int(np.count_nonzero(differences == 0)),
                "folds_combined_lower": int(np.count_nonzero(differences < 0)),
            })
    return comparisons


def run(config_path: Path = PROJECT_ROOT / "config.json") -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    phase3_config = config["bonn_smoke"]
    phase4_config = config["bonn_full"]
    out = PROJECT_ROOT / "outputs" / "bonn_full"
    out.mkdir(parents=True, exist_ok=True)

    # Assert the full-run numerical choices are unchanged from the frozen smoke pipeline.
    frozen_keys = ("delay_dim", "svd_rank", "koopman_coordinate_count", "coordinate_source", "reference_mode", "finite_p", "finite_p_weight_mode")
    mismatches = {key: (phase3_config[key], phase4_config[key]) for key in frozen_keys if phase3_config[key] != phase4_config[key]}
    if mismatches:
        raise ValueError(f"Phase 4 parameters differ from frozen Phase 3 pipeline: {mismatches}")
    if float(phase4_config["sampling_rate_hz"]) != float(data_config["sampling_rate_hz"]):
        raise ValueError("Phase 4 sampling rate differs from the fixed Bonn data configuration")
    if phase4_config["primary_deviation"] != "chebyshev":
        raise ValueError("primary deviation must remain Chebyshev")

    snapshot = {
        "project": config["project"],
        "paper": config["project"]["paper"],
        "data": data_config,
        "frozen_phase3_koopman_parameters": {key: phase3_config[key] for key in frozen_keys},
        "phase4": phase4_config,
        "feature_definitions": config["features"],
        "software_versions": {
            "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "imbalanced_learn": imblearn.__version__,
        },
        "protocol": {
            "validation_unit": "segment-level",
            "patient_identity_available": "not established; no patient-independent claim",
            "all_folds_shared_across_feature_sets_and_models": True,
            "scalers_fit_within_training_folds_only": True,
            "hyperparameter_search": False,
            "feature_selection": False,
            "band_decomposition": False,
        },
    }
    _write_json(out / "config_snapshot.json", snapshot)

    files_by_label = discover_bonn_files(data_config)
    expected_count = int(data_config["expected_segments_per_class"])
    for label in ("D", "E"):
        if len(files_by_label.get(label, [])) != expected_count:
            raise RuntimeError(f"expected {expected_count} {label} segments, found {len(files_by_label.get(label, []))}")

    feature_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    sample_signals: dict[str, np.ndarray] = {}
    process_results: dict[str, dict[str, Any]] = {}
    fs = float(phase4_config["sampling_rate_hz"])
    for label in ("D", "E"):
        state = "interictal" if label == "D" else "ictal"
        for path in files_by_label[label]:
            sample_id = path.stem.upper()
            try:
                signal = load_bonn_segment(path)
                sample_signals[sample_id] = signal
                result = _process_signal(signal, phase4_config, fs)
            except Exception as exc:
                signal = None
                result = {
                    "segment_length": 0,
                    "H_shape": "",
                    "singular_values": "",
                    "linearity_error": None,
                    "spectral_radius": None,
                    "nan_inf_occurred": False,
                    "svd_attempted": False,
                    "svd_succeeded": False,
                    "eig_attempted": False,
                    "eig_succeeded": False,
                    "pipeline_success": False,
                    "error_reason": f"{type(exc).__name__}: {exc}",
                    "features": {},
                    "D_cheb": None,
                    "D_finite_p": None,
                    "Y": None,
                }
            process_results[sample_id] = result
            metric_rows.append(_metric_row(sample_id, label, state, path, result, phase4_config))
            feature_rows.append({
                "sample_id": sample_id,
                "label": label,
                "state": state,
                "linearity_error": result["linearity_error"] if result["linearity_error"] is not None else "",
                "spectral_radius": result["spectral_radius"] if result["spectral_radius"] is not None else "",
                **{name: result["features"].get(name, "") for name in KOOPMAN_FEATURES + FINITE_P_FEATURES + RAW_FEATURES},
            })
        print(f"Extracted {label}: {len(files_by_label[label])} segments")

    metric_columns = [
        "sample_id", "label", "state", "file_path", "segment_length", "delay_dim", "svd_rank",
        "coordinate_count", "coordinate_source", "reference_mode", "H_shape", "singular_values",
        "linearity_error", "spectral_radius", "nan_inf_occurred", "svd_attempted", "svd_succeeded",
        "eig_attempted", "eig_succeeded", "pipeline_success", "error_reason",
    ]
    _write_csv(out / "features.csv", feature_rows, ["sample_id", "label", "state", "linearity_error", "spectral_radius", *KOOPMAN_FEATURES, *FINITE_P_FEATURES, *RAW_FEATURES])
    _write_csv(out / "per_sample_metrics.csv", metric_rows, metric_columns)

    success_rows = [row for row in metric_rows if bool(row["pipeline_success"])]
    class_success = {label: sum(row["label"] == label and bool(row["pipeline_success"]) for row in metric_rows) for label in ("D", "E")}
    numerical_summary = {
        "expected_samples": 200,
        "successful_samples": len(success_rows),
        "failed_samples": len(metric_rows) - len(success_rows),
        "successful_by_label": class_success,
        "nan_inf_samples": sum(bool(row["nan_inf_occurred"]) for row in metric_rows),
        "svd_attempted": sum(bool(row["svd_attempted"]) for row in metric_rows),
        "svd_failures": sum(bool(row["svd_attempted"]) and not bool(row["svd_succeeded"]) for row in metric_rows),
        "eig_attempted": sum(bool(row["eig_attempted"]) for row in metric_rows),
        "eig_failures": sum(bool(row["eig_attempted"]) and not bool(row["eig_succeeded"]) for row in metric_rows),
        "pipeline_failures_before_svd": sum(not bool(row["svd_attempted"]) for row in metric_rows),
        "failed_sample_reasons": [{"sample_id": row["sample_id"], "label": row["label"], "reason": row["error_reason"]} for row in metric_rows if not bool(row["pipeline_success"])],
    }
    _write_json(out / "numerical_audit.json", numerical_summary)
    if len(success_rows) != 200:
        _write_json(out / "phase4_run_status.json", {"status": "FAILED", "reason": "not all 200 segments produced complete features; CV/statistics were not run on a silently reduced dataset", "numerical_audit": numerical_summary})
        raise RuntimeError(f"Phase 4 requires all 200 complete feature rows; got {len(success_rows)}")

    linearity_rows = _linearity_summary(metric_rows)
    _write_csv(out / "linearity_spectral_summary.csv", linearity_rows)
    _plot_linearity(metric_rows, out / "Figure_BonnFull_linearity_distribution.png")

    values_koopman = _group_feature_values(feature_rows, KOOPMAN_FEATURES)
    koopman_tests = feature_test_rows(values_koopman, KOOPMAN_FEATURES)
    _write_csv(out / "statistical_tests.csv", koopman_tests)
    values_raw = _group_feature_values(feature_rows, RAW_FEATURES)
    raw_tests = feature_test_rows(values_raw, RAW_FEATURES)
    _write_csv(out / "raw_statistical_tests.csv", raw_tests)
    _plot_koopman_distributions(feature_rows, out / "Figure_BonnFull_Koopman_feature_distributions.png")
    _plot_dmax_dstd(feature_rows, out / "Figure_BonnFull_Dmax_Dstd.png")
    _plot_raw_distributions(feature_rows, out / "Figure_BonnFull_raw_feature_distributions.png")

    scale_report = _scale_control(sample_signals, process_results, phase4_config)
    _write_json(out / "scale_invariance_control.json", scale_report)

    # Shared folds are generated once; each feature set/model sees identical sample IDs.
    cv_rows, oof_rows, split_manifest, classifier_summary = _cv_evaluation(
        feature_rows,
        phase4_config["feature_sets"],
        phase4_config["classifiers"],
        phase4_config["cv"],
        out,
    )
    paired = _paired_model_comparison(cv_rows)
    _write_csv(out / "classifier_pairwise_comparison.csv", paired)
    _plot_classifier_summary(classifier_summary, out / "Figure_BonnFull_classifier_comparison.png")
    _write_json(out / "phase4_run_status.json", {
        "status": "ANALYSIS COMPLETE",
        "validation_unit": "segment-level",
        "n_segments": 200,
        "folds": len(split_manifest),
        "oof_predictions": len(oof_rows),
        "numerical_audit": numerical_summary,
        "scale_control_success": all(item.get("success", False) for item in scale_report["samples"].values()),
    })
    print(f"Full Bonn outputs: {out}")
    print(f"Feature extraction: {len(success_rows)}/200; SVD failures={numerical_summary['svd_failures']}; eig failures={numerical_summary['eig_failures']}")
    print(f"Completed {len(cv_rows)} fold-model-feature-set evaluations; saved {len(oof_rows)} OOF rows")
    return out


if __name__ == "__main__":
    with warnings.catch_warnings():
        # Keep third-party deprecation warnings visible; suppress only convergence warnings? None are suppressed.
        warnings.simplefilter("default")
        run()
