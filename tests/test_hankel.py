import numpy as np
import pytest

from src.hankel import build_hankel, truncated_svd


def test_build_hankel_example():
    actual = build_hankel([1, 2, 3, 4, 5], delay_dim=3)
    expected = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=np.float64)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (3, 3)
    assert actual.dtype == np.float64


def test_delay_dimension_one_has_one_row():
    actual = build_hankel([4, 5, 6], delay_dim=1)
    np.testing.assert_array_equal(actual, [[4, 5, 6]])
    assert actual.shape == (1, 3)


def test_delay_dimension_equal_to_signal_length_has_one_column():
    actual = build_hankel([4, 5, 6], delay_dim=3)
    np.testing.assert_array_equal(actual, [[4], [5], [6]])
    assert actual.shape == (3, 1)


@pytest.mark.parametrize("delay_dim", [0, -1, 4, 1.5, True])
def test_invalid_delay_dimension_raises(delay_dim):
    with pytest.raises((TypeError, ValueError)):
        build_hankel([1, 2, 3], delay_dim=delay_dim)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_signal_raises(bad_value):
    with pytest.raises(ValueError, match="NaN or Inf"):
        build_hankel([1, bad_value, 3], delay_dim=2)


def test_non_vector_signal_raises():
    with pytest.raises(ValueError):
        build_hankel([[1, 2], [3, 4]], delay_dim=2)


def test_truncated_svd_full_rank_reconstructs_hankel():
    H = build_hankel([1, 2, 3, 4, 5], delay_dim=3)
    full_rank = min(H.shape)
    U_r, S_r, Vh_r, singular_values = truncated_svd(H, rank=full_rank)
    reconstructed = U_r @ np.diag(S_r) @ Vh_r
    np.testing.assert_allclose(reconstructed, H, rtol=1e-13, atol=1e-13)
    assert singular_values.shape == (full_rank,)
    np.testing.assert_allclose(singular_values, S_r)


def test_truncated_svd_rejects_rank_above_matrix_limit():
    with pytest.raises(ValueError, match="cannot exceed"):
        truncated_svd(np.ones((2, 3)), rank=3)
