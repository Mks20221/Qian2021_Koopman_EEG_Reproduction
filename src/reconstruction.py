"""Reconstruct trajectories in the fitted finite-dimensional coordinates."""

from __future__ import annotations

import numpy as np


def _real_finite_array(values: object, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional; got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError(f"{name} must contain real numeric values")
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must not contain NaN or Inf")
    return array


def _square_operator(A: object) -> np.ndarray:
    operator = _real_finite_array(A, "A", ndim=2)
    if operator.shape[0] < 1 or operator.shape[0] != operator.shape[1]:
        raise ValueError(f"A must be a nonempty square matrix; got {operator.shape}")
    return operator


def reconstruct_linear_trajectory(A: object, y0: object, n_steps: int) -> np.ndarray:
    """Generate y_0,...,y_(n_steps-1), including the initial state y0."""
    operator = _square_operator(A)
    initial_state = _real_finite_array(y0, "y0", ndim=1)
    if initial_state.size != operator.shape[0]:
        raise ValueError("y0 length must match A dimension")
    if isinstance(n_steps, (bool, np.bool_)) or not isinstance(
        n_steps, (int, np.integer)
    ):
        raise TypeError("n_steps must be an integer")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1 to include y0")

    trajectory = np.empty((initial_state.size, int(n_steps)), dtype=np.float64)
    trajectory[:, 0] = initial_state
    for index in range(1, int(n_steps)):
        try:
            with np.errstate(over="raise", invalid="raise"):
                next_state = operator @ trajectory[:, index - 1]
        except FloatingPointError as exc:
            raise FloatingPointError(f"trajectory overflow at step {index}") from exc
        if not np.all(np.isfinite(next_state)):
            raise FloatingPointError(f"trajectory became non-finite at step {index}")
        trajectory[:, index] = next_state
    return trajectory


def one_step_prediction(A: object, Y: object) -> np.ndarray:
    """Predict each next state from observed columns: A @ Y[:, :-1]."""
    operator = _square_operator(A)
    coordinates = _real_finite_array(Y, "Y", ndim=2)
    if coordinates.shape[0] != operator.shape[0] or coordinates.shape[1] < 2:
        raise ValueError("Y must have shape [A dimension, N] with N >= 2")
    try:
        with np.errstate(over="raise", invalid="raise"):
            prediction = operator @ coordinates[:, :-1]
    except FloatingPointError as exc:
        raise FloatingPointError("one-step prediction overflowed") from exc
    if not np.all(np.isfinite(prediction)):
        raise FloatingPointError("one-step prediction returned a non-finite value")
    return np.asarray(prediction, dtype=np.float64)
