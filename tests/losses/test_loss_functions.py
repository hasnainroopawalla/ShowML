from showml.losses import MeanSquaredError
import numpy as np
from numpy.testing import assert_almost_equal


def test_mse_objective_simple() -> None:
    y = np.array([3, -0.5, 2, 7])
    z = np.array([2.5, 0.0, 2, 8])
    MSE = MeanSquaredError()
    assert MSE.objective(y, z) == 0.375


def test_mse_objective_complex() -> None:
    y = np.array([0.5, 1, -1, 1, 7, -6])
    z = np.array([0, 2, -1, 2, 8, -5])
    MSE = MeanSquaredError()
    assert MSE.objective(y, z) == 0.7083333333333334


def test_mse_objective_no_error() -> None:
    y = np.array([0, 1, 3, 2, 5.78])
    z = np.array([0, 1, 3, 2, 5.78])
    MSE = MeanSquaredError()
    assert MSE.objective(y, z) == 0.0


def test_mse_gradient() -> None:
    X = np.array([[0.5, 1], [-1, 1], [7, -6], [6, 8], [1, 1]])
    y = np.array([0.5, 1, -1, 1, -6])
    z = np.array([0, 2, -1, 2, -5])
    MSE = MeanSquaredError()
    dw, db = MSE.parameter_gradient(X, y, z)
    assert_almost_equal(dw, [1.15, 1.9])
    assert db == 0.5


def test_mse_gradient_zero() -> None:
    X = np.array([[0.5, 1], [-1, 1], [7, -6], [6, 8], [1, 1]])
    y = np.array([0, 1, 3, 2, 5.78])
    z = np.array([0, 1, 3, 2, 5.78])
    MSE = MeanSquaredError()
    dw, db = MSE.parameter_gradient(X, y, z)
    assert_almost_equal(dw, [0.0, 0.0])
    assert db == 0.0
