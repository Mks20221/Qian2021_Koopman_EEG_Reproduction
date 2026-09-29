import numpy as np
import pytest

from src.koopman import (
    build_intrinsic_coordinates,
    eigenvalue_diagnostics,
    fit_linear_dynamics,
    sort_eigenvalues_by_magnitude,
)


def known_rotation_system(n_steps=200):
    theta = 0.2
    radius = 0.95
    A_true = radius * np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )
    Y = np.empty((2, n_steps), dtype=np.float64)
    Y[:, 0] = [1.0, 0.0]
    for index in range(1, n_steps):
        Y[:, index] = A_true @ Y[:, index - 1]
    return A_true, Y, radius, theta


def test_known_linear_system_recovers_operator_orientation_and_spectrum():
    A_true, Y, radius, theta = known_rotation_system()
    fit = fit_linear_dynamics(Y)

    relative_operator_error = np.linalg.norm(fit["A"] - A_true, ord="fro") / np.linalg.norm(
        A_true, ord="fro"
    )
    assert relative_operator_error < 1e-12
    assert fit["linearity_error"] < 1e-12

    expected = np.array(
        [radius * np.exp(1j * theta), radius * np.exp(-1j * theta)],
        dtype=np.complex128,
    )
    actual = fit["eigvals"]
    pairing_error = min(
        np.max(np.abs(actual - expected)),
        np.max(np.abs(actual - expected[::-1])),
    )
    assert pairing_error < 1e-12
    assert fit["A"].dtype == np.float64
    assert fit["eigvals"].dtype == np.complex128


def test_build_intrinsic_coordinates_supports_v_and_sig_mav():
    U_r = np.eye(2, dtype=np.float64)
    S_r = np.array([3.0, 2.0])
    Vh_r = np.array([[0.6, 0.8], [-0.8, 0.6]])

    np.testing.assert_array_equal(build_intrinsic_coordinates(U_r, S_r, Vh_r, "V"), Vh_r)
    np.testing.assert_array_equal(
        build_intrinsic_coordinates(U_r, S_r, Vh_r, "SigmaV"),
        np.diag(S_r) @ Vh_r,
    )
    with pytest.raises(ValueError, match="source"):
        build_intrinsic_coordinates(U_r, S_r, Vh_r, "U")


def test_eigenvalue_diagnostics_marks_zero_continuous_time_value_invalid():
    values = np.array([0.0, 0.95 * np.exp(0.2j)], dtype=np.complex128)
    diagnostics = eigenvalue_diagnostics(values, dt=0.01)

    np.testing.assert_array_equal(diagnostics["continuous_time_valid"], [False, True])
    assert np.isnan(diagnostics["mu"][0].real)
    assert np.isnan(diagnostics["mu"][0].imag)
    assert np.isnan(diagnostics["frequency_hz"][0])
    assert np.isnan(diagnostics["growth_rate"][0])
    assert np.all(np.isfinite(diagnostics["magnitude"]))
    assert np.all(np.isfinite(diagnostics["phase"]))
    expected_mu = (np.log(0.95) + 0.2j) / 0.01
    assert diagnostics["mu"][1] == pytest.approx(expected_mu)
    assert diagnostics["frequency_hz"][1] == pytest.approx(0.2 / (0.01 * 2.0 * np.pi))
    assert diagnostics["growth_rate"][1] == pytest.approx(np.log(0.95) / 0.01)


def test_eigenvalue_magnitude_sort_is_diagnostic_only():
    values = np.array([0.2 + 0j, 0.9j, -0.5 + 0j])
    original = values.copy()
    order = sort_eigenvalues_by_magnitude(values)
    np.testing.assert_array_equal(order, [1, 2, 0])
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("bad_Y", [np.array([[1.0]]), np.array([[1.0, np.nan]])])
def test_fit_linear_dynamics_rejects_invalid_trajectory(bad_Y):
    with pytest.raises(ValueError):
        fit_linear_dynamics(bad_Y)
