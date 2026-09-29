"""Explicit Bonn D/E file discovery, loading, and raw-data audit."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np


def discover_bonn_files(data_config: dict[str, Any]) -> dict[str, list[Path]]:
    """Discover fixed D_F/Fxxx and E_S/Sxxx text segments, case-insensitive by extension."""
    root = Path(data_config["bonn_root"])
    if not root.is_dir():
        raise FileNotFoundError(f"Bonn data root does not exist: {root}")
    found: dict[str, list[Path]] = {}
    for class_name in ("interictal", "ictal"):
        class_info = data_config["classes"][class_name]
        label = str(class_info["label"])
        folder = root / str(class_info["folder"])
        prefix = re.escape(str(class_info["filename_prefix"]))
        id_pattern = re.compile(rf"^{prefix}\d{{3}}$", re.IGNORECASE)
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected Bonn class folder is missing: {folder}")
        by_stem: dict[str, Path] = {}
        for path in folder.iterdir():
            if not path.is_file() or path.suffix.lower() != ".txt":
                continue
            if not id_pattern.fullmatch(path.stem):
                continue
            key = path.stem.upper()
            if key in by_stem:
                raise ValueError(
                    f"Duplicate segment ID {key} in {folder}: {by_stem[key].name} and {path.name}"
                )
            by_stem[key] = path
        found[label] = sorted(by_stem.values(), key=lambda p: p.stem.upper())
    return found


def load_bonn_segment(path: str | Path) -> np.ndarray:
    """Load one numeric single-channel Bonn text segment as a flat float64 vector."""
    source = Path(path)
    values = np.loadtxt(source, dtype=np.float64, ndmin=1)
    signal = np.asarray(values, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"expected a single-channel vector, got shape {signal.shape}")
    if signal.size == 0:
        raise ValueError("segment is empty")
    return signal


def _amplitude_summary(arrays: list[np.ndarray]) -> dict[str, float | int | None]:
    finite_parts = [np.asarray(a)[np.isfinite(a)] for a in arrays]
    finite_parts = [part for part in finite_parts if part.size]
    if not finite_parts:
        return {
            "finite_sample_count": 0,
            "minimum_amplitude": None,
            "maximum_amplitude": None,
            "mean_amplitude_pooled": None,
            "std_amplitude_pooled_ddof0": None,
        }
    pooled = np.concatenate(finite_parts)
    return {
        "finite_sample_count": int(pooled.size),
        "minimum_amplitude": float(np.min(pooled)),
        "maximum_amplitude": float(np.max(pooled)),
        "mean_amplitude_pooled": float(np.mean(pooled, dtype=np.float64)),
        "std_amplitude_pooled_ddof0": float(np.std(pooled, ddof=0, dtype=np.float64)),
    }


def audit_bonn_files(
    files_by_label: dict[str, list[Path]], sampling_rate_hz: float
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read and audit the D/E source files; return audit JSON data and loaded signals.

    The audit covers all discovered D/E segments. The Koopman pipeline should
    still be run only on the separately selected smoke subset.
    """
    loaded: dict[str, np.ndarray] = {}
    class_summaries: dict[str, Any] = {}
    all_arrays: list[np.ndarray] = []
    all_errors: list[dict[str, str]] = []

    for label in ("D", "E"):
        paths = files_by_label.get(label, [])
        lengths: list[int] = []
        arrays: list[np.ndarray] = []
        errors: list[dict[str, str]] = []
        nan_count = 0
        inf_count = 0
        for path in paths:
            try:
                signal = load_bonn_segment(path)
            except Exception as exc:  # Preserve audit errors rather than hiding an unreadable file.
                item = {"sample_id": path.stem, "path": str(path), "error": f"{type(exc).__name__}: {exc}"}
                errors.append(item)
                all_errors.append({**item, "label": label})
                continue
            loaded[str(path.resolve())] = signal
            arrays.append(signal)
            all_arrays.append(signal)
            lengths.append(int(signal.size))
            nan_count += int(np.count_nonzero(np.isnan(signal)))
            inf_count += int(np.count_nonzero(np.isinf(signal)))
        length_counts: dict[str, int] = {}
        for length in lengths:
            key = str(length)
            length_counts[key] = length_counts.get(key, 0) + 1
        class_summaries[label] = {
            "discovered_segment_count": len(paths),
            "read_success_count": len(arrays),
            "segment_length_distribution": dict(sorted(length_counts.items(), key=lambda item: int(item[0]))),
            "minimum_segment_length": min(lengths) if lengths else None,
            "maximum_segment_length": max(lengths) if lengths else None,
            "nan_count": nan_count,
            "inf_count": inf_count,
            **_amplitude_summary(arrays),
            "read_errors": errors,
        }
    all_lengths = [int(array.size) for array in all_arrays]
    total_nan = sum(int(np.count_nonzero(np.isnan(a))) for a in all_arrays)
    total_inf = sum(int(np.count_nonzero(np.isinf(a))) for a in all_arrays)
    audit = {
        "dataset": "Bonn",
        "source_scope": "all discovered D_F/Fxxx and E_S/Sxxx text segments; no other Bonn classes",
        "sampling_rate_hz": float(sampling_rate_hz),
        "extension_handling": ".txt and .TXT accepted case-insensitively",
        "class_mapping": {"D": "D_F/Fxxx; interictal", "E": "E_S/Sxxx; ictal"},
        "classes": class_summaries,
        "overall": {
            "discovered_segment_count": sum(len(files_by_label.get(label, [])) for label in ("D", "E")),
            "read_success_count": len(all_arrays),
            "segment_length_distribution": {
                key: sum(summary["segment_length_distribution"].get(key, 0) for summary in class_summaries.values())
                for key in sorted(
                    {key for summary in class_summaries.values() for key in summary["segment_length_distribution"]},
                    key=int,
                )
            },
            "minimum_segment_length": min(all_lengths) if all_lengths else None,
            "maximum_segment_length": max(all_lengths) if all_lengths else None,
            "nan_count": total_nan,
            "inf_count": total_inf,
            **_amplitude_summary(all_arrays),
            "read_error_count": len(all_errors),
            "read_errors": all_errors,
        },
    }
    return audit, loaded
