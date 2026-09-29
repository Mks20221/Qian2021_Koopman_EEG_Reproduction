import numpy as np

from src.koopman import fit_linear_dynamics
from src.reconstruction import one_step_prediction, reconstruct_linear_trajectory


def test_one_dimensional_geometric_trajectory_includes_initial_state():
    actual = reconstruct_linear_trajectory([[0.9]], [1.0], n_steps=4)
    np.testing.assert_allclose(actual, [[1.0, 0.9, 0.81, 0.729]], rtol=1e-14, atol=1e-14)


def test_known_linear_system_reconstruction_matches_observed_trajectory():
    theta = 0.2
    radius = 0.95
    A_true = radius * np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )
    Y = np.empty((2, 200), dtype=np.float64)
    Y[:, 0] = [1.0, 0.0]
    for index in range(1, Y.shape[1]):
        Y[:, index] = A_true @ Y[:, index - 1]

    fit = fit_linear_dynamics(Y)
    reconstructed = reconstruct_linear_trajectory(fit["A"], Y[:, 0], Y.shape[1])
    max_error = np.max(np.abs(reconstructed - Y))
    assert max_error < 1e-12


def test_one_step_prediction_matches_next_observed_states():
    A = np.array([[0.9, 0.0], [0.0, 0.8]])
    Y = np.array([[1.0, 0.9, 0.81], [2.0, 1.6, 1.28]])
    prediction = one_step_prediction(A, Y)
    np.testing.assert_allclose(prediction, Y[:, 1:], rtol=1e-14, atol=1e-14)
