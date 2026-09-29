"""Explicit, auditable numerical components for the Qian & Chou reproduction."""

from .deviation import (
    chebyshev_distance,
    finite_p_weighted_distance,
    make_eigenvalue_weights,
    trajectory_deviation,
)
from .hankel import build_hankel, truncated_svd
from .koopman import (
    build_intrinsic_coordinates,
    eigenvalue_diagnostics,
    fit_linear_dynamics,
    sort_eigenvalues_by_magnitude,
)
from .reconstruction import one_step_prediction, reconstruct_linear_trajectory

__all__ = [
    "build_hankel",
    "truncated_svd",
    "build_intrinsic_coordinates",
    "fit_linear_dynamics",
    "eigenvalue_diagnostics",
    "sort_eigenvalues_by_magnitude",
    "reconstruct_linear_trajectory",
    "one_step_prediction",
    "finite_p_weighted_distance",
    "chebyshev_distance",
    "trajectory_deviation",
    "make_eigenvalue_weights",
]
