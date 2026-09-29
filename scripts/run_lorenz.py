"""Run the pre-registered Phase 2 Lorenz reproduction diagnostics."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deviation import make_eigenvalue_weights, trajectory_deviation
from src.hankel import build_hankel, truncated_svd
from src.koopman import build_intrinsic_coordinates, eigenvalue_diagnostics, fit_linear_dynamics
from src.lorenz import simulate_l1_l2_l1
from src.plots import (
    plot_coordinate_source_comparison,
    plot_deviation,
    plot_embedding_linearity,
    plot_embedding_comparison,
    plot_intrinsic_trajectory,
    plot_lorenz_3d,
    plot_lorenz_signals,
)
from src.reconstruction import reconstruct_linear_trajectory


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    raise TypeError(f"cannot encode {type(value).__name__} as JSON")


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _coordinate_diagnostics(Y: np.ndarray, dt: float, delay_dim: int, source: str) -> dict[str, object]:
    fit = fit_linear_dynamics(Y)
    eigen = eigenvalue_diagnostics(fit["eigvals"], dt)
    return {
        "source": source,
        "delay_dim": int(delay_dim),
        "linearity_error": float(fit["linearity_error"]),
        "coordinate_rms": float(np.sqrt(np.mean(np.square(Y), dtype=np.float64))),
        "max_abs": float(np.max(np.abs(Y))),
        "spectral_radius": float(np.max(eigen["magnitude"])),
        "fit": fit,
    }


def _relative_frobenius(error: np.ndarray, target: np.ndarray) -> float:
    error_norm = float(np.linalg.norm(error, ord="fro"))
    target_norm = float(np.linalg.norm(target, ord="fro"))
    return error_norm / target_norm if target_norm else error_norm


def _short_horizon_errors(A: np.ndarray, Y: np.ndarray, horizon: int, stride: int) -> dict[str, object]:
    # horizon is the number of transitions; each comparison includes y0 and y_horizon.
    last_start = Y.shape[1] - horizon - 1
    starts = np.arange(0, last_start + 1, stride, dtype=int)
    errors: list[float] = []
    diverged = 0
    for start in starts:
        actual = Y[:, start : start + horizon + 1]
        try:
            predicted = reconstruct_linear_trajectory(A, actual[:, 0], horizon + 1)
            error = _relative_frobenius(predicted - actual, actual)
            if not np.isfinite(error):
                diverged += 1
            else:
                errors.append(error)
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError):
            diverged += 1
    return {
        "horizon_transitions": int(horizon),
        "stride_samples": int(stride),
        "windows_attempted": int(starts.size),
        "windows_finite": len(errors),
        "windows_diverged": diverged,
        "relative_frobenius_mean": float(np.mean(errors)) if errors else None,
        "relative_frobenius_median": float(np.median(errors)) if errors else None,
        "relative_frobenius_max": float(np.max(errors)) if errors else None,
    }


def run(config_path: Path = PROJECT_ROOT / "config.json") -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    lorenz_config = config["lorenz"]
    delay_dims = [int(value) for value in lorenz_config["delay_dim_candidates"]]
    rank = int(config["koopman"]["svd_rank"])
    dt = float(lorenz_config["sample_interval_s"])
    out_dir = PROJECT_ROOT / "outputs" / "lorenz"
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "project": config["project"],
        "lorenz": lorenz_config,
        "koopman": {key: config["koopman"][key] for key in ("svd_rank", "coordinate_source", "coordinate_source_provenance", "coordinate_selection_criterion", "coordinate_selection_numeric_implementation")},
        "deviation": config["deviation"],
        "phase2_protocol": {
            "analysis_signal": "x(t) only",
            "rank_fixed": rank,
            "delay_dim_candidates_fixed": delay_dims,
            "transition_coordinate_source": lorenz_config["transition_coordinate_source"],
            "transition_delay_dim": int(lorenz_config["transition_delay_dim"]),
            "reference_mode": "origin",
            "finite_p": int(lorenz_config["transition_distance_p"]),
            "finite_p_weights": "abs(eigenvalue), implementation choice",
        },
    }
    _write_json(out_dir / "config_snapshot.json", snapshot)

    simulation = simulate_l1_l2_l1(
        initial_condition=lorenz_config["initial_condition"],
        burn_in_s=float(lorenz_config["burn_in_duration_s"]),
        segment_duration_s=float(lorenz_config["integration_duration_per_segment_s"]),
        dt=dt,
        parameters={key: {name: float(value) for name, value in lorenz_config[key].items() if name in ("sigma", "rho", "beta")} for key in ("L1", "L2")},
        rtol=float(lorenz_config["solver_rtol"]),
        atol=float(lorenz_config["solver_atol"]),
        max_step=float(lorenz_config["max_step_s"]),
    )
    times = simulation["time_s"]
    states = simulation["states_xyz"]
    boundaries = simulation["boundary_times_s"]
    labels = simulation["segment_labels"]
    _write_csv(
        out_dir / "lorenz_simulation.csv",
        [
            {"time_s": f"{times[i]:.8f}", "x": f"{states[i,0]:.12g}", "y": f"{states[i,1]:.12g}", "z": f"{states[i,2]:.12g}", "regime": labels[i]}
            for i in range(len(times))
        ],
    )
    plot_lorenz_3d(times, states, boundaries, out_dir / "Figure_01_Lorenz_L1_L2_L1_3D.png")
    plot_lorenz_signals(times, states, boundaries, out_dir / "Figure_02_Lorenz_signals.png")
    plot_lorenz_3d(times, states, boundaries, out_dir / "Figure_Lorenz_L1_L2_L1_3D.png")

    # The dimension comparison uses only x(t) from the first post-burn-in L1 segment.
    stable_l1_x = simulation["segment_samples"]["L1_first"][:, 0]
    embedding_rows: list[dict[str, object]] = []
    coordinate_rows: list[dict[str, object]] = []
    embeddings: dict[int, dict[str, object]] = {}
    for delay_dim in delay_dims:
        H = build_hankel(stable_l1_x, delay_dim)
        U_r, S_r, Vh_r, singular_values = truncated_svd(H, rank)
        candidate_info: dict[str, object] = {}
        for source in ("V", "SigmaV"):
            Y = build_intrinsic_coordinates(U_r, S_r, Vh_r, source=source)
            diagnostic = _coordinate_diagnostics(Y, dt, delay_dim, source)
            candidate_info[source] = {"Y": Y, **diagnostic}
            coordinate_rows.append({
                "delay_dim": delay_dim,
                "rank": rank,
                "coordinate_source": source,
                "linearity_error": diagnostic["linearity_error"],
                "coordinate_rms": diagnostic["coordinate_rms"],
                "max_abs": diagnostic["max_abs"],
                "spectral_radius": diagnostic["spectral_radius"],
            })
        positive_singular = singular_values[singular_values > 0]
        full_condition = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else None
        retained_condition = float(S_r[0] / S_r[-1]) if S_r[-1] > 0 else None
        embedding_rows.append({
            "delay_dim": delay_dim,
            "rank": rank,
            "hankel_rows": int(H.shape[0]),
            "hankel_columns": int(H.shape[1]),
            "singular_values": json.dumps(singular_values.tolist(), separators=(",", ":")),
            "full_condition_number": full_condition,
            "retained_rank_condition_number": retained_condition,
            "smallest_positive_singular_value": float(positive_singular[-1]) if positive_singular.size else 0.0,
            "linearity_error_V": candidate_info["V"]["linearity_error"],
            "spectral_radius_V": candidate_info["V"]["spectral_radius"],
        })
        embeddings[delay_dim] = {"H": H, "U_r": U_r, "S_r": S_r, "Vh_r": Vh_r, **candidate_info}
    _write_csv(out_dir / "embedding_metrics.csv", embedding_rows)
    _write_csv(out_dir / "coordinate_source_metrics.csv", coordinate_rows)
    plot_embedding_linearity(embedding_rows, out_dir / "Figure_03_embedding_dimension_linearity.png")
    plot_embedding_linearity(embedding_rows, out_dir / "Figure_Lorenz_linearity_error.png")

    low_dim, high_dim = delay_dims[0], delay_dims[-1]
    plot_intrinsic_trajectory(embeddings[low_dim]["V"]["Y"], low_dim, out_dir / f"Figure_04_intrinsic_V_delay_{low_dim}.png")
    plot_intrinsic_trajectory(embeddings[high_dim]["V"]["Y"], high_dim, out_dir / f"Figure_05_intrinsic_V_delay_{high_dim}.png")
    plot_embedding_comparison(
        embeddings[low_dim]["V"]["Y"],
        low_dim,
        embeddings[high_dim]["V"]["Y"],
        high_dim,
        out_dir / "Figure_Lorenz_embedding_comparison.png",
    )
    comparison_dim = int(lorenz_config["transition_delay_dim"])
    comparison = embeddings[comparison_dim]
    source_metrics = {source: {key: float(comparison[source][key]) for key in ("linearity_error", "coordinate_rms", "max_abs")} for source in ("V", "SigmaV")}
    plot_coordinate_source_comparison(comparison["V"]["Y"], comparison["SigmaV"]["Y"], comparison_dim, source_metrics, out_dir / f"Figure_06_coordinate_sources_delay_{comparison_dim}.png")

    # Refit the same fixed-rank pipeline on the full L1 -> L2 -> L1 x observation.
    transition_delay = int(lorenz_config["transition_delay_dim"])
    transition_H = build_hankel(states[:, 0], transition_delay)
    U_r, S_r, Vh_r, transition_singular = truncated_svd(transition_H, rank)
    Y = build_intrinsic_coordinates(U_r, S_r, Vh_r, source=str(lorenz_config["transition_coordinate_source"]))
    fit = fit_linear_dynamics(Y)
    eig_diag = eigenvalue_diagnostics(fit["eigvals"], dt)
    lambda_weights = make_eigenvalue_weights(fit["eigvals"], mode="abs")
    D_cheb = trajectory_deviation(Y, reference_mode="origin", distance_mode="chebyshev")
    p_value = float(lorenz_config["transition_distance_p"])
    D_finite_p = trajectory_deviation(
        Y,
        reference_mode="origin",
        distance_mode="paper_literal_finite_p",
        weights=lambda_weights,
        p=p_value,
    )
    anchor_times = times[0] + (np.arange(Y.shape[1], dtype=np.float64) + (transition_delay - 1) / 2.0) * dt
    plot_deviation(anchor_times, D_cheb, boundaries, "Chebyshev, origin reference", out_dir / "Figure_07_D_chebyshev.png")
    plot_deviation(anchor_times, D_finite_p, boundaries, f"finite p={p_value:g}, |lambda| weights, origin reference", out_dir / "Figure_08_D_finite_p.png")

    deviation_rows = []
    for name, values in (("D_chebyshev", D_cheb), ("D_finite_p_abs_lambda", D_finite_p)):
        for segment_name, mask in (
            ("L1_first", anchor_times < boundaries[0]),
            ("L2", (anchor_times >= boundaries[0]) & (anchor_times < boundaries[1])),
            ("L1_final", anchor_times >= boundaries[1]),
        ):
            selected = values[mask]
            deviation_rows.append({
                "deviation_method": name,
                "segment": segment_name,
                "n_windows": int(selected.size),
                "mean": float(np.mean(selected)),
                "median": float(np.median(selected)),
                "std": float(np.std(selected, ddof=0)),
                "max": float(np.max(selected)),
                "center_time_min_s": float(np.min(anchor_times[mask])),
                "center_time_max_s": float(np.max(anchor_times[mask])),
            })
    _write_csv(out_dir / "deviation_summary.csv", deviation_rows)

    one_step_relative = float(fit["linearity_error"])
    one_step_rmse = float(np.sqrt(np.mean(np.square(fit["Y2"] - fit["Y2_hat"]), dtype=np.float64)))
    short_horizon = _short_horizon_errors(
        fit["A"], Y,
        int(lorenz_config["short_horizon_samples"]),
        int(lorenz_config["short_horizon_stride_samples"]),
    )
    diagnostics = {
        "hankel_shape": list(transition_H.shape),
        "rank": rank,
        "delay_dim": transition_delay,
        "coordinate_source": str(lorenz_config["transition_coordinate_source"]),
        "linearity_error_relative_frobenius": one_step_relative,
        "one_step_rmse": one_step_rmse,
        "spectral_radius": float(np.max(eig_diag["magnitude"])),
        "eigenvalue_magnitudes": eig_diag["magnitude"],
        "eigenvalue_phases_rad": eig_diag["phase"],
        "continuous_time_frequency_hz": eig_diag["frequency_hz"],
        "continuous_time_growth_rate": eig_diag["growth_rate"],
        "short_horizon_free_run": short_horizon,
        "finite_value_checks": {
            "H": bool(np.all(np.isfinite(transition_H))),
            "Y": bool(np.all(np.isfinite(Y))),
            "A": bool(np.all(np.isfinite(fit["A"]))),
            "eigenvalues": bool(np.all(np.isfinite(fit["eigvals"]))),
            "D_chebyshev": bool(np.all(np.isfinite(D_cheb))),
            "D_finite_p": bool(np.all(np.isfinite(D_finite_p))),
        },
        "time_anchor": "center of each Hankel delay window",
        "time_anchor_formula": "t_j=t0+(j+(delay_dim-1)/2)*dt",
        "dynamical_deviation_candidates": {
            "reference": "origin; paper-motivated, numerical reference construction unspecified",
            "chebyshev": "unweighted infinity norm candidate",
            "finite_p": f"p={p_value:g}, weights=abs(eigvals); implementation choice / paper-unspecified numerical handling",
        },
        "transition_singular_values": transition_singular,
    }
    _write_json(out_dir / "reconstruction_metrics.json", diagnostics)
    _write_json(out_dir / "lorenz_run_summary.json", {
        "total_samples": int(states.shape[0]),
        "samples_per_segment": int(simulation["samples_per_segment"]),
        "sampling_rate_hz": 1.0 / dt,
        "burn_in_s": float(lorenz_config["burn_in_duration_s"]),
        "boundary_times_s": boundaries,
        "boundary_states_xyz": simulation["boundary_states_xyz"],
        "L1_parameters": lorenz_config["L1"],
        "L2_parameters": lorenz_config["L2"],
        "embedding_linearity_error_V": {str(row["delay_dim"]): row["linearity_error_V"] for row in embedding_rows},
        "transition_diagnostics": diagnostics,
    })
    print(f"Lorenz Phase 2 outputs: {out_dir}")
    print("Embedding linearity error (V): " + ", ".join(f"{row['delay_dim']}={row['linearity_error_V']:.6g}" for row in embedding_rows))
    print(f"Transition linearity error={one_step_relative:.6g}; spectral radius={diagnostics['spectral_radius']:.6g}")
    print(f"Free-run 1 s windows: finite={short_horizon['windows_finite']}/{short_horizon['windows_attempted']}, mean relative error={short_horizon['relative_frobenius_mean']}")
    return out_dir


if __name__ == "__main__":
    run()
