"""Lorenz integration and continuous L1 -> L2 -> L1 regime switching."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.integrate import solve_ivp


PAPER_LORENZ_PARAMETERS = {
    "L1": {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
    "L2": {"sigma": 10.0, "rho": 28.0, "beta": 4.0 / 3.0},
}


def lorenz_rhs(
    time_s: float,
    state: object,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> np.ndarray:
    """Evaluate the classical three-variable Lorenz vector field."""
    del time_s  # The autonomous vector field does not depend explicitly on time.
    values = np.asarray(state)
    if values.shape != (3,) or not np.issubdtype(values.dtype, np.number):
        raise ValueError("state must be a real numeric vector with shape [3]")
    if np.iscomplexobj(values):
        raise TypeError("state must be real")
    values = np.asarray(values, dtype=np.float64)
    parameters = np.asarray([sigma, rho, beta], dtype=np.float64)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(parameters)):
        raise ValueError("state and Lorenz parameters must be finite")

    x, y, z = values
    sigma_value, rho_value, beta_value = parameters
    derivative = np.array(
        [
            sigma_value * (y - x),
            x * (rho_value - z) - y,
            x * y - beta_value * z,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(derivative)):
        raise FloatingPointError("Lorenz vector field returned a non-finite derivative")
    return derivative


def _positive_finite_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or np.iscomplexobj(value):
        raise TypeError(f"{name} must be a positive finite real scalar")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _duration_steps(duration_s: float, dt: float, name: str) -> int:
    ratio = duration_s / dt
    steps = int(round(ratio))
    if steps < 1 or not np.isclose(ratio, steps, rtol=0.0, atol=1e-10):
        raise ValueError(f"{name} must be an integer number of dt samples")
    return steps


def _state_vector(state: object, name: str) -> np.ndarray:
    values = np.asarray(state)
    if values.shape != (3,) or not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{name} must have shape [3] and contain real numeric values")
    if np.iscomplexobj(values):
        raise TypeError(f"{name} must be real")
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must not contain NaN or Inf")
    return values


def integrate_lorenz_steps(
    initial_state: object,
    n_steps: int,
    dt: float,
    parameters: Mapping[str, float],
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    max_step: float | None = None,
) -> np.ndarray:
    """Integrate from an initial state and return states after each dt step.

    The result has shape [n_steps, 3] and excludes ``initial_state`` itself.
    Consequently, the final row is the exact state passed into the next
    regime segment when the integration is split at a boundary.
    """
    start = _state_vector(initial_state, "initial_state")
    if isinstance(n_steps, (bool, np.bool_)) or not isinstance(n_steps, (int, np.integer)):
        raise TypeError("n_steps must be a positive integer")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    dt_value = _positive_finite_scalar(dt, "dt")
    rtol_value = _positive_finite_scalar(rtol, "rtol")
    atol_value = _positive_finite_scalar(atol, "atol")
    max_step_value = dt_value if max_step is None else _positive_finite_scalar(max_step, "max_step")
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping with sigma, rho, and beta")
    try:
        sigma, rho, beta = (float(parameters[key]) for key in ("sigma", "rho", "beta"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("parameters must contain numeric sigma, rho, and beta") from exc
    if not np.all(np.isfinite([sigma, rho, beta])):
        raise ValueError("Lorenz parameters must be finite")

    sample_times = dt_value * np.arange(1, int(n_steps) + 1, dtype=np.float64)
    try:
        solution = solve_ivp(
            lambda time_s, values: lorenz_rhs(
                time_s, values, sigma=sigma, rho=rho, beta=beta
            ),
            (0.0, float(n_steps) * dt_value),
            start,
            method="RK45",
            t_eval=sample_times,
            rtol=rtol_value,
            atol=atol_value,
            max_step=max_step_value,
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        raise RuntimeError("Lorenz numerical integration raised an error") from exc
    if not solution.success:
        raise RuntimeError(f"Lorenz integration failed: {solution.message}")
    if solution.y.shape != (3, int(n_steps)):
        raise RuntimeError(
            f"Lorenz integration returned shape {solution.y.shape}; "
            f"expected (3, {int(n_steps)})"
        )
    samples = np.asarray(solution.y.T, dtype=np.float64)
    if not np.all(np.isfinite(samples)):
        raise FloatingPointError("Lorenz integration returned NaN or Inf")
    return samples


def simulate_l1_l2_l1(
    *,
    initial_condition: object = (1.0, 1.0, 1.0),
    burn_in_s: float = 20.0,
    segment_duration_s: float = 50.0,
    dt: float = 0.01,
    parameters: Mapping[str, Mapping[str, float]] | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    max_step: float | None = None,
) -> dict[str, object]:
    """Generate one continuous L1 -> L2 -> L1 trajectory after L1 burn-in.

    Each analyzed regime contains ``segment_duration_s / dt`` samples. The
    boundary state is the first sample of the next regime and is reused as
    that regime's initial condition. The burn-in is not included in returned
    observations.
    """
    initial = _state_vector(initial_condition, "initial_condition")
    burn_in_value = _positive_finite_scalar(burn_in_s, "burn_in_s")
    segment_duration = _positive_finite_scalar(segment_duration_s, "segment_duration_s")
    dt_value = _positive_finite_scalar(dt, "dt")
    burn_steps = _duration_steps(burn_in_value, dt_value, "burn_in_s")
    segment_steps = _duration_steps(segment_duration, dt_value, "segment_duration_s")
    parameters = PAPER_LORENZ_PARAMETERS if parameters is None else parameters
    if not all(name in parameters for name in ("L1", "L2")):
        raise ValueError("parameters must define both L1 and L2")

    burn_trajectory = integrate_lorenz_steps(
        initial,
        burn_steps,
        dt_value,
        parameters["L1"],
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    current_state = burn_trajectory[-1].copy()
    segment_names = ("L1_first", "L2", "L1_final")
    segment_parameter_names = ("L1", "L2", "L1")
    segment_samples: dict[str, np.ndarray] = {}
    transition_states: dict[str, np.ndarray] = {}

    for name, parameter_name in zip(segment_names, segment_parameter_names):
        integrated_steps = integrate_lorenz_steps(
            current_state,
            segment_steps,
            dt_value,
            parameters[parameter_name],
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        # Samples are t=0, dt, ..., (n-1)dt relative to this regime's start.
        # integrated_steps[-1] is the exact state at its duration boundary;
        # that boundary state starts the following regime and is not duplicated.
        samples = np.empty((segment_steps, 3), dtype=np.float64)
        samples[0] = current_state
        if segment_steps > 1:
            samples[1:] = integrated_steps[:-1]
        segment_samples[name] = samples
        current_state = integrated_steps[-1].copy()
        transition_states[name] = current_state.copy()

    all_states = np.vstack([segment_samples[name] for name in segment_names])
    times = np.arange(all_states.shape[0], dtype=np.float64) * dt_value
    labels = np.concatenate(
        [np.full(segment_steps, name, dtype="U16") for name in segment_names]
    )
    boundary_times = np.array(
        [segment_duration, 2.0 * segment_duration], dtype=np.float64
    )
    boundary_states = np.vstack(
        [segment_samples["L2"][0], segment_samples["L1_final"][0]]
    )

    if not np.all(np.isfinite(all_states)):
        raise FloatingPointError("combined Lorenz trajectory contains NaN or Inf")
    return {
        "time_s": times,
        "states_xyz": all_states,
        "segment_labels": labels,
        "segment_samples": segment_samples,
        "boundary_times_s": boundary_times,
        "boundary_states_xyz": boundary_states,
        "burn_in_end_state_xyz": burn_trajectory[-1].copy(),
        "burn_in_samples": int(burn_steps),
        "samples_per_segment": int(segment_steps),
        "parameters": parameters,
    }
