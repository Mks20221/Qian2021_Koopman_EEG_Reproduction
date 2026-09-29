"""Publication-style Lorenz figures for the Phase 2 reproduction."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


_SEGMENT_COLORS = {"L1_first": "#2563eb", "L2": "#dc2626", "L1_final": "#16a34a"}


def _save(fig: plt.Figure, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_lorenz_3d(
    time_s: np.ndarray,
    states_xyz: np.ndarray,
    boundaries_s: Sequence[float],
    path: str | Path,
) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    n = len(time_s)
    for index, (name, color) in enumerate(_SEGMENT_COLORS.items()):
        left = index * n // 3
        right = (index + 1) * n // 3
        segment = states_xyz[left:right]
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, lw=0.55, label=name)
    for boundary in boundaries_s:
        sample_index = int(round(boundary / (time_s[1] - time_s[0])))
        point = states_xyz[sample_index]
        ax.scatter(*point, color="black", marker="o", s=30, depthshade=False)
    ax.set(title="Lorenz trajectory: L1 → L2 → L1", xlabel="x", ylabel="y", zlabel="z")
    ax.legend(loc="upper left")
    _save(fig, path)


def plot_lorenz_signals(
    time_s: np.ndarray,
    states_xyz: np.ndarray,
    boundaries_s: Sequence[float],
    path: str | Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for index, (name, color) in enumerate(zip(("x", "y", "z"), ("#2563eb", "#dc2626", "#16a34a"))):
        axes[index].plot(time_s, states_xyz[:, index], color=color, lw=0.65)
        axes[index].set_ylabel(name)
        axes[index].grid(alpha=0.2)
        for boundary in boundaries_s:
            axes[index].axvline(boundary, color="black", ls="--", lw=0.8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Lorenz observations across the L1 → L2 → L1 transition")
    fig.tight_layout()
    _save(fig, path)


def plot_embedding_linearity(rows: Sequence[dict[str, object]], path: str | Path) -> None:
    dims = np.asarray([int(row["delay_dim"]) for row in rows])
    errors = np.asarray([float(row["linearity_error_V"]) for row in rows])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(dims, errors, "o-", color="#1d4ed8", lw=1.8)
    ax.set(xlabel="Delay embedding dimension", ylabel="Relative one-step linearity error", title="Embedding dimension and intrinsic-space linearity")
    ax.set_xticks(dims)
    ax.grid(alpha=0.25)
    _save(fig, path)


def plot_intrinsic_trajectory(
    coordinates: np.ndarray,
    delay_dim: int,
    path: str | Path,
) -> None:
    if coordinates.shape[0] < 3:
        raise ValueError("at least three intrinsic coordinates are required")
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(coordinates[0], coordinates[1], coordinates[2], color="#2563eb", lw=0.7)
    ax.set(xlabel="$y_1$", ylabel="$y_2$", zlabel="$y_3$", title=f"First three V coordinates, delay_dim={delay_dim}")
    _save(fig, path)


def plot_embedding_comparison(
    low_coordinates: np.ndarray,
    low_dim: int,
    high_coordinates: np.ndarray,
    high_dim: int,
    path: str | Path,
) -> None:
    """Show the first three V coordinates at the pre-registered low/high dims."""
    fig = plt.figure(figsize=(13, 5.8))
    for panel, (coordinates, delay_dim) in enumerate(
        ((low_coordinates, low_dim), (high_coordinates, high_dim)), start=1
    ):
        if coordinates.shape[0] < 3:
            raise ValueError("at least three intrinsic coordinates are required")
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        ax.plot(coordinates[0], coordinates[1], coordinates[2], color="#2563eb", lw=0.65)
        ax.set(
            xlabel="$y_1$",
            ylabel="$y_2$",
            zlabel="$y_3$",
            title=f"V coordinates, delay_dim={delay_dim}",
        )
    fig.suptitle("Intrinsic-coordinate comparison; no rotation or rescaling")
    _save(fig, path)


def plot_coordinate_source_comparison(
    V: np.ndarray,
    SigmaV: np.ndarray,
    delay_dim: int,
    metrics: dict[str, dict[str, float]],
    path: str | Path,
) -> None:
    if min(V.shape[0], SigmaV.shape[0]) < 3:
        raise ValueError("at least three intrinsic coordinates are required")
    fig = plt.figure(figsize=(13, 5.8))
    for panel, (name, coordinates) in enumerate((("V", V), ("SigmaV", SigmaV)), start=1):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        ax.plot(coordinates[0], coordinates[1], coordinates[2], color="#2563eb", lw=0.65)
        diagnostic = metrics[name]
        ax.set(xlabel="$y_1$", ylabel="$y_2$", zlabel="$y_3$", title=f"{name}: err={diagnostic['linearity_error']:.4g}, RMS={diagnostic['coordinate_rms']:.4g}, max|y|={diagnostic['max_abs']:.4g}")
    fig.suptitle(f"Coordinate-source candidates, delay_dim={delay_dim}; scale is not normalized")
    _save(fig, path)


def plot_deviation(
    anchor_times_s: np.ndarray,
    deviation: np.ndarray,
    boundaries_s: Sequence[float],
    method_label: str,
    path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(anchor_times_s, deviation, color="#7c3aed", lw=0.65)
    for boundary in boundaries_s:
        ax.axvline(boundary, color="black", ls="--", lw=0.9)
    ax.set(xlabel="Hankel-window center time (s)", ylabel="$D(t)$", title=f"Lorenz reconstructed-state deviation: {method_label}")
    ax.grid(alpha=0.2)
    _save(fig, path)
