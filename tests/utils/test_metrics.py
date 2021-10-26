from showml.utils.metrics import calculate_training_error, calculate_r2_score
import numpy as np
from numpy.testing import assert_almost_equal


def test_calculate_training_error_simple() -> None:
    y = np.array([2, 3, 4, 5])
    z = np.array([1, 2, 3, 4])
    assert_almost_equal(calculate_training_error(y, z), [-1, -1, -1, -1])


def test_calculate_training_error_zero() -> None:
    y = np.array([7.8, 5.6, 3.2, 1, -8.6])
    z = np.array([7.8, 5.6, 3.2, 1, -8.6])
    assert_almost_equal(calculate_training_error(y, z), [0, 0, 0, 0, 0])


def test_calculate_r2_score_simple() -> None:
    y = np.array([4, 3, 2, 1])
    z = np.array([1, 2, 3, 4])
    assert calculate_r2_score(y, z) == -3.0


def test_calculate_r2_score_float_1D() -> None:
    y = np.array([3, -0.5, 2, 7])
    z = np.array([2.5, 0.0, 2, 8])
    assert calculate_r2_score(y, z) == 0.9486081370449679


def test_calculate_r2_score_float_2D() -> None:
    z = np.array([[0, 2], [-1, 2], [8, -5]])
    y = np.array([[0.5, 1], [-1, 1], [7, -6]])
    assert calculate_r2_score(y, z) == 0.9512661251791686
