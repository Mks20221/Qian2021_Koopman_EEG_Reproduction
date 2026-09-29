"""Run the fixed 5-D + 5-E Bonn numerical smoke test (no classifier/CV)."""

from __future__ import annotations

import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import audit_bonn_files, discover_bonn_files
from src.deviation import make_eigenvalue_weights, trajectory_deviation
from src.features import raw_summary_features, summary_features
from src.hankel import build_hankel, truncated_svd
from src.koopman import build_intrinsic_coordinates, eigenvalue_diagnostics, fit_linear_dynamics


FEATURE_COLUMNS = [
    "Dmax_cheb", "Dmin_cheb", "Dmean_cheb", "Dstd_cheb", "Dpower_cheb", "Dentropy_cheb",
    "Dmax_fp", "Dmin_fp", "Dmean_fp", "Dstd_fp", "Dpower_fp", "Dentropy_fp",
    "raw_mean", "raw_std", "raw_power", "raw_entropy",
]


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


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if not rows and not fieldnames:
        raise ValueError(f"fieldnames are required for empty CSV {path}")
    names = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_raw_examples(samples: list[dict[str, Any]], fs: float, path: Path) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=False)
    for ax, item in zip(axes.flat, samples):
        signal = item.get("signal")
        if signal is None:
            ax.text(0.5, 0.5, "signal load failed", ha="center", va="center", transform=ax.transAxes)
        else:
            time_s = np.arange(signal.size, dtype=np.float64) / fs
            ax.plot(time_s, signal, lw=0.55, color="#1d4ed8")
        ax.set_title(f"{item['sample_id']} | {item['label']} | {item['state']}", fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.18)
    fig.suptitle("Bonn smoke subset: raw EEG (seeded random sample)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_deviation_examples(samples: list[dict[str, Any]], fs: float, delay_dim: int, method: str, path: Path) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=False)
    key = "D_cheb" if method == "chebyshev" else "D_finite_p"
    for ax, item in zip(axes.flat, samples):
        deviation = item.get(key)
        if deviation is None:
            ax.text(0.5, 0.5, f"pipeline failed\n{item.get('error_reason', '')[:80]}", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        else:
            center_time = (np.arange(deviation.size, dtype=np.float64) + (delay_dim - 1) / 2.0) / fs
            ax.plot(center_time, deviation, lw=0.65, color="#7c3aed")
        ax.set_title(f"{item['sample_id']} | {item['label']}", fontsize=9)
        ax.set_xlabel("Hankel-window center time (s)")
        ax.set_ylabel("D(t)")
        ax.grid(alpha=0.18)
    title = "Chebyshev D(t), primary" if method == "chebyshev" else "Finite-p D(t), p=20 and abs(lambda) weights"
    fig.suptitle(f"Bonn smoke subset: {title}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_intrinsic_examples(samples: list[dict[str, Any]], path: Path) -> list[str]:
    representatives: list[dict[str, Any]] = []
    for label in ("D", "E"):
        successful = [item for item in samples if item["label"] == label and item.get("Y") is not None]
        representatives.extend(successful[:2])
    fig = plt.figure(figsize=(12, 9))
    if not representatives:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No successful intrinsic trajectories", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        for index, item in enumerate(representatives, start=1):
            ax = fig.add_subplot(2, 2, index, projection="3d")
            Y = item["Y"]
            ax.plot(Y[0], Y[1], Y[2], lw=0.7, color="#2563eb" if item["label"] == "D" else "#dc2626")
            ax.set(xlabel="$y_1$", ylabel="$y_2$", zlabel="$y_3$", title=f"{item['sample_id']} | {item['label']} | V")
    fig.suptitle("Bonn intrinsic trajectories (first two successful random samples per class)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return [str(item["sample_id"]) for item in representatives]


def _plot_feature_table(rows: list[dict[str, object]], path: Path) -> None:
    columns = ["sample_id", "label", "Dmean_cheb", "Dstd_cheb", "Dmax_cheb", "Dmean_fp", "Dstd_fp", "raw_std", "raw_power"]
    cell_text = []
    for row in rows:
        cells: list[str] = []
        for name in columns:
            value = row.get(name)
            if value in (None, ""):
                cells.append("—")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.4g}")
            else:
                cells.append(str(value))
        cell_text.append(cells)
    fig, ax = plt.subplots(figsize=(14, 5.3))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.45)
    for (row_index, _column), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_facecolor("#dbeafe")
            cell.set_text_props(weight="bold")
        elif row_index > 0:
            cell.set_facecolor("#eff6ff" if cell_text[row_index - 1][1] == "D" else "#fef2f2")
    ax.set_title("Bonn smoke summary features (10 seeded samples; primary D uses Chebyshev)", pad=18)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _process_sample(item: dict[str, Any], cfg: dict[str, Any], fs: float, entropy_bins: int) -> dict[str, Any]:
    signal = item.get("signal")
    row: dict[str, Any] = {
        "sample_id": item["sample_id"], "label": item["label"], "state": item["state"],
        "file_path": item["file_path"], "segment_length": 0 if signal is None else int(signal.size),
        "delay_dim": int(cfg["delay_dim"]), "svd_rank": int(cfg["svd_rank"]),
        "coordinate_count": int(cfg["koopman_coordinate_count"]), "coordinate_source": cfg["coordinate_source"],
        "H_shape": "", "singular_values": "", "linearity_error": "", "spectral_radius": "",
        "nan_inf_occurred": False, "eig_attempted": False, "eig_succeeded": False,
        "pipeline_success": False, "error_reason": "",
    }
    if signal is None:
        row["error_reason"] = item.get("load_error", "segment was not readable during audit")
        row["nan_inf_occurred"] = True
        item["error_reason"] = row["error_reason"]
        return row
    row["nan_inf_occurred"] = bool(not np.all(np.isfinite(signal)))
    if row["nan_inf_occurred"]:
        row["error_reason"] = "input segment contains NaN or Inf; core pipeline requires finite data"
        item["error_reason"] = row["error_reason"]
        try:
            item["raw_features"] = raw_summary_features(signal[np.isfinite(signal)], bins=entropy_bins) if np.any(np.isfinite(signal)) else None
        except Exception:
            item["raw_features"] = None
        return row
    try:
        item["raw_features"] = raw_summary_features(signal, bins=entropy_bins)
        H = build_hankel(signal, int(cfg["delay_dim"]))
        row["H_shape"] = f"{H.shape[0]}x{H.shape[1]}"
        U_r, S_r, Vh_r, all_singular_values = truncated_svd(H, int(cfg["svd_rank"]))
        row["singular_values"] = json.dumps(all_singular_values.tolist(), separators=(",", ":"))
        Y = build_intrinsic_coordinates(U_r, S_r, Vh_r, source=str(cfg["coordinate_source"]))
        item["Y"] = Y
        row["eig_attempted"] = True
        fit = fit_linear_dynamics(Y)
        row["linearity_error"] = float(fit["linearity_error"])
        row["eig_succeeded"] = True
        eigen = eigenvalue_diagnostics(fit["eigvals"], dt=1.0 / fs)
        row["spectral_radius"] = float(np.max(eigen["magnitude"]))
        weights = make_eigenvalue_weights(fit["eigvals"], mode=str(cfg["finite_p_weight_mode"]))
        D_cheb = trajectory_deviation(Y, reference_mode=str(cfg["reference_mode"]), distance_mode="chebyshev")
        D_finite_p = trajectory_deviation(
            Y,
            reference_mode=str(cfg["reference_mode"]),
            distance_mode="paper_literal_finite_p",
            weights=weights,
            p=float(cfg["finite_p"]),
        )
        item["D_cheb"] = D_cheb
        item["D_finite_p"] = D_finite_p
        row["nan_inf_occurred"] = bool(
            row["nan_inf_occurred"]
            or not np.all(np.isfinite(H))
            or not np.all(np.isfinite(Y))
            or not np.all(np.isfinite(fit["A"]))
            or not np.all(np.isfinite(fit["eigvals"]))
            or not np.all(np.isfinite(D_cheb))
            or not np.all(np.isfinite(D_finite_p))
        )
        if row["nan_inf_occurred"]:
            raise FloatingPointError("NaN/Inf detected in an intermediate pipeline result")
        cheb = summary_features(D_cheb, bins=entropy_bins)
        finite_p = summary_features(D_finite_p, bins=entropy_bins)
        feature_values = {
            "Dmax_cheb": cheb["max"], "Dmin_cheb": cheb["min"],
            "Dmean_cheb": cheb["mean"], "Dstd_cheb": cheb["std"],
            "Dpower_cheb": cheb["power"], "Dentropy_cheb": cheb["entropy"],
            "Dmax_fp": finite_p["max"], "Dmin_fp": finite_p["min"],
            "Dmean_fp": finite_p["mean"], "Dstd_fp": finite_p["std"],
            "Dpower_fp": finite_p["power"], "Dentropy_fp": finite_p["entropy"],
            **item["raw_features"],
        }
        item["features"] = feature_values
        row["pipeline_success"] = True
    except Exception as exc:
        row["error_reason"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, FloatingPointError) or "NaN" in str(exc) or "Inf" in str(exc):
            row["nan_inf_occurred"] = True
        item["error_reason"] = row["error_reason"]
        item["traceback"] = traceback.format_exc()
        if row["linearity_error"] == "":
            item["Y"] = None
        return row
    return row


def run(config_path: Path = PROJECT_ROOT / "config.json") -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    smoke_cfg = config["bonn_smoke"]
    out_dir = PROJECT_ROOT / "outputs" / "bonn_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = {
        "project": config["project"],
        "data": data_cfg,
        "bonn_smoke": smoke_cfg,
        "reproducibility": config["reproducibility"],
        "features": config["features"],
        "phase3_scope": "10 randomly selected segments only for Koopman pipeline; no CV/classifiers/band decomposition",
    }
    _write_json(out_dir / "config_snapshot.json", config_snapshot)

    files_by_label = discover_bonn_files(data_cfg)
    audit, loaded_signals = audit_bonn_files(files_by_label, float(data_cfg["sampling_rate_hz"]))
    _write_json(out_dir / "data_audit.json", audit)
    print("Bonn D/E audit summary:")
    for label in ("D", "E"):
        class_audit = audit["classes"][label]
        print(
            f"  {label}: discovered={class_audit['discovered_segment_count']}, "
            f"read={class_audit['read_success_count']}, "
            f"lengths={class_audit['segment_length_distribution']}, "
            f"NaN={class_audit['nan_count']}, Inf={class_audit['inf_count']}, "
            f"amplitude=[{class_audit['minimum_amplitude']}, {class_audit['maximum_amplitude']}]"
        )

    rng = np.random.default_rng(int(smoke_cfg["random_seed"]))
    per_class = int(smoke_cfg["samples_per_class"])
    selected: list[dict[str, Any]] = []
    for label, state in (("D", "interictal"), ("E", "ictal")):
        candidates = files_by_label.get(label, [])
        if len(candidates) < per_class:
            raise RuntimeError(f"not enough {label} segments: found {len(candidates)}, need {per_class}")
        chosen_indices = rng.choice(len(candidates), size=per_class, replace=False)
        for order, index in enumerate(chosen_indices, start=1):
            path = candidates[int(index)]
            resolved = str(path.resolve())
            item: dict[str, Any] = {
                "sample_id": path.stem.upper(), "label": label, "state": state,
                "file_path": str(path), "relative_path": str(path.relative_to(Path(data_cfg["bonn_root"]))),
                "selection_order_within_class": order,
                "signal": loaded_signals.get(resolved), "Y": None,
            }
            if item["signal"] is None:
                item["load_error"] = next(
                    (error["error"] for error in audit["classes"][label]["read_errors"] if error["sample_id"].upper() == item["sample_id"]),
                    "segment could not be loaded during audit",
                )
            selected.append(item)

    manifest_rows = [
        {
            "sample_id": item["sample_id"], "label": item["label"], "state": item["state"],
            "relative_path": item["relative_path"], "selection_order_within_class": item["selection_order_within_class"],
            "random_seed": int(smoke_cfg["random_seed"]),
        }
        for item in selected
    ]
    _write_csv(out_dir / "sample_manifest.csv", manifest_rows)

    fs = float(data_cfg["sampling_rate_hz"])
    entropy_bins = int(config["features"]["entropy_histogram_bins"])
    metrics_rows = [_process_sample(item, smoke_cfg, fs, entropy_bins) for item in selected]
    feature_rows: list[dict[str, object]] = []
    for item in selected:
        values = item.get("features", {})
        row: dict[str, object] = {
            "sample_id": item["sample_id"], "label": item["label"],
            "delay_dim": int(smoke_cfg["delay_dim"]), "svd_rank": int(smoke_cfg["svd_rank"]),
            "linearity_error": "", "spectral_radius": "",
        }
        metric = next(result for result in metrics_rows if result["sample_id"] == item["sample_id"] and result["label"] == item["label"])
        row["linearity_error"] = metric["linearity_error"]
        row["spectral_radius"] = metric["spectral_radius"]
        row.update({column: values.get(column, "") for column in FEATURE_COLUMNS})
        feature_rows.append(row)
    _write_csv(
        out_dir / "features.csv", feature_rows,
        ["sample_id", "label", "delay_dim", "svd_rank", "linearity_error", "spectral_radius", *FEATURE_COLUMNS],
    )
    metric_columns = [
        "sample_id", "label", "state", "file_path", "segment_length", "delay_dim", "svd_rank",
        "coordinate_count", "coordinate_source", "H_shape", "singular_values", "linearity_error",
        "spectral_radius", "nan_inf_occurred", "eig_attempted", "eig_succeeded", "pipeline_success", "error_reason",
    ]
    _write_csv(out_dir / "per_sample_metrics.csv", metrics_rows, metric_columns)

    _plot_raw_examples(selected, fs, out_dir / "Figure_BonnSmoke_raw_examples.png")
    _plot_deviation_examples(selected, fs, int(smoke_cfg["delay_dim"]), "chebyshev", out_dir / "Figure_BonnSmoke_D_chebyshev_examples.png")
    _plot_deviation_examples(selected, fs, int(smoke_cfg["delay_dim"]), "finite_p", out_dir / "Figure_BonnSmoke_D_finitep_examples.png")
    intrinsic_ids = _plot_intrinsic_examples(selected, out_dir / "Figure_BonnSmoke_intrinsic_examples.png")
    _plot_feature_table(feature_rows, out_dir / "Figure_BonnSmoke_feature_table.png")

    successful = [row for row in metrics_rows if row["pipeline_success"]]
    failures = [row for row in metrics_rows if not row["pipeline_success"]]
    class_success = {label: sum(row["label"] == label and row["pipeline_success"] for row in metrics_rows) for label in ("D", "E")}
    separation_notes: dict[str, Any] = {}
    cheb_feature_names = ("Dmax_cheb", "Dmin_cheb", "Dmean_cheb", "Dstd_cheb", "Dpower_cheb", "Dentropy_cheb")
    qualitative_nonoverlap: list[str] = []
    for feature_name in cheb_feature_names:
        feature_groups: dict[str, list[float]] = {}
        for label in ("D", "E"):
            values = [float(row[feature_name]) for row in feature_rows if row["label"] == label and row.get(feature_name) not in (None, "")]
            feature_groups[label] = values
            separation_notes.setdefault(feature_name, {})[label] = {
                "n": len(values), "minimum": min(values) if values else None,
                "maximum": max(values) if values else None,
                "mean": float(np.mean(values)) if values else None,
            }
        d_values = feature_groups["D"]
        e_values = feature_groups["E"]
        if d_values and e_values and (max(d_values) < min(e_values) or max(e_values) < min(d_values)):
            qualitative_nonoverlap.append(feature_name)
    raw_feature_ranges: dict[str, Any] = {}
    for feature_name in ("raw_mean", "raw_std", "raw_power", "raw_entropy"):
        raw_feature_ranges[feature_name] = {}
        for label in ("D", "E"):
            values = [float(row[feature_name]) for row in feature_rows if row["label"] == label and row.get(feature_name) not in (None, "")]
            raw_feature_ranges[feature_name][label] = {
                "n": len(values), "minimum": min(values) if values else None,
                "maximum": max(values) if values else None,
                "mean": float(np.mean(values)) if values else None,
            }
    enough_successes = (
        len(successful) >= int(smoke_cfg["minimum_success_count_for_smoke_pass"])
        and all(class_success[label] >= int(smoke_cfg["minimum_success_per_class_for_smoke_pass"]) for label in ("D", "E"))
    )
    status = "FAILED"
    if enough_successes:
        status = "PASS" if qualitative_nonoverlap else "SMOKE PASS, SEPARATION UNCERTAIN"
    run_summary = {
        "status": status,
        "selected_count": len(selected), "successful_count": len(successful), "failed_count": len(failures),
        "success_by_label": class_success,
        "failed_samples": [{"sample_id": row["sample_id"], "label": row["label"], "error_reason": row["error_reason"]} for row in failures],
        "nan_inf_sample_count": sum(bool(row["nan_inf_occurred"]) for row in metrics_rows),
        "eigendecomposition_attempted_count": sum(bool(row["eig_attempted"]) for row in metrics_rows),
        "eigendecomposition_failure_count": sum(bool(row["eig_attempted"]) and not bool(row["eig_succeeded"]) for row in metrics_rows),
        "pipeline_failures_before_eigendecomposition": sum(not bool(row["eig_attempted"]) for row in metrics_rows),
        "intrinsic_plot_sample_ids": intrinsic_ids,
        "primary_feature_ranges": separation_notes,
        "raw_feature_ranges": raw_feature_ranges,
        "qualitative_nonoverlap_primary_features": qualitative_nonoverlap,
        "separation_assessment": "descriptive ranges within the fixed random smoke subset only; no significance test or classifier",
    }
    _write_json(out_dir / "smoke_summary.json", run_summary)
    errors = [float(row["linearity_error"]) for row in successful if row["linearity_error"] not in (None, "")]
    print(f"Selected IDs: {', '.join(item['sample_id'] for item in selected)}")
    print(f"Pipeline successes D={class_success['D']}/5, E={class_success['E']}/5; failures={len(failures)}")
    print(f"NaN/Inf samples={run_summary['nan_inf_sample_count']}; eig failures={run_summary['eigendecomposition_failure_count']}")
    if errors:
        print(f"linearity_error range=[{min(errors):.6g}, {max(errors):.6g}]")
    print(f"Smoke status: {run_summary['status']}; outputs: {out_dir}")
    return out_dir


if __name__ == "__main__":
    run()
