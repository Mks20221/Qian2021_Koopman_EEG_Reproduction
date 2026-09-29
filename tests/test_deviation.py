import numpy as np
import pytest

from src.deviation import (
    chebyshev_distance,
    finite_p_weighted_distance,
    make_eigenvalue_weights,
    trajectory_deviation,
)


def test_chebyshev_distance():
    assert chebyshev_distance([1.0, 2.0], [0.0, 0.0]) == 2.0


def test_finite_p_weighted_distance_l2():
    actual = finite_p_weighted_distance([1.0, 2.0], [0.0, 0.0], [1.0, 1.0], p=2)
    assert actual == pytest.approx(np.sqrt(5.0))


def test_all_distances_are_zero_at_reference():
    x = [2.0, -3.0]
    reference = [2.0, -3.0]
    assert chebyshev_distance(x, reference) == 0.0
    assert finite_p_weighted_distance(x, reference, [1.0, 2.0], p=50) == 0.0


def test_finite_p_approaches_chebyshev_as_p_increases():
    values = [
        finite_p_weighted_distance([1.0, 2.0], [0.0, 0.0], [1.0, 1.0], p=p)
        for p in (2, 10, 50)
    ]
    assert values[0] > values[1] > values[2]
    assert values[2] == pytest.approx(2.0, abs=1e-12)


def test_finite_p_is_stable_for_large_values_and_high_p():
    actual = finite_p_weighted_distance(
        [1e100, 2e100], [0.0, 0.0], [1.0, 1.0], p=50
    )
    assert actual == pytest.approx(2e100, rel=1e-12)


def test_complex_weights_are_rejected():
    with pytest.raises(TypeError, match="real"):
        finite_p_weighted_distance([1.0, 2.0], [0.0, 0.0], [1.0 + 1j, 1.0], p=2)


def test_negative_weights_are_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        finite_p_weighted_distance([1.0, 2.0], [0.0, 0.0], [1.0, -1.0], p=2)


def test_non_finite_weights_and_nonpositive_p_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        finite_p_weighted_distance([1.0], [0.0], [np.inf], p=2)
    with pytest.raises(ValueError, match="positive"):
        finite_p_weighted_distance([1.0], [0.0], [1.0], p=0)


def test_trajectory_deviation_from_origin():
    Y = np.array([[1.0, 0.0], [0.0, 2.0]])
    actual = trajectory_deviation(Y, distance_mode="chebyshev", reference_mode="origin")
    np.testing.assert_array_equal(actual, [1.0, 2.0])


def test_trajectory_deviation_from_trajectory_centroid():
    Y = np.array([[1.0, 3.0], [2.0, 4.0]])
    actual = trajectory_deviation(
        Y, distance_mode="chebyshev", reference_mode="trajectory_centroid"
    )
    np.testing.assert_array_equal(actual, [1.0, 1.0])


def test_trajectory_deviation_from_explicit_reference():
    Y = np.array([[1.0, 2.0], [3.0, 4.0]])
    actual = trajectory_deviation(
        Y,
        reference=[0.0, 1.0],
        distance_mode="chebyshev",
        reference_mode="explicit",
    )
    np.testing.assert_array_equal(actual, [2.0, 3.0])


def test_eigenvalue_weight_choice_is_absolute_magnitude_only():
    values = np.array([1j, -2.0 + 0j])
    np.testing.assert_array_equal(make_eigenvalue_weights(values, mode="abs"), [1.0, 2.0])
    with pytest.raises(ValueError, match="only mode='abs'"):
        make_eigenvalue_weights(values, mode="real")


def test_training_reference_centroid_is_not_implemented_in_phase_one():
    with pytest.raises(NotImplementedError, match="training-fold"):
        trajectory_deviation(
            np.ones((2, 3)), reference_mode="training_reference_centroid"
        )
