from showml.losses.loss_functions import MeanSquareError
import numpy as np
from numpy.testing import assert_almost_equal


"""
Mean Square Error
"""
def test_objective_simple() -> None:
    y = np.array([3, -0.5, 2, 7])
    z = np.array([2.5, 0.0, 2, 8])
    MSE = MeanSquareError()
    assert MSE.objective(y, z) == 0.1875

def test_objective_complex() -> None:
    y = np.array([0.5, 1, -1, 1, 7, -6])
    z = np.array([0, 2, -1, 2, 8, -5])
    MSE = MeanSquareError()
    assert MSE.objective(y, z) == 0.3541666666666667

def test_objective_no_error() -> None:
    y = np.array([0, 1, 3, 2, 5.78])
    z = np.array([0, 1, 3, 2, 5.78])
    MSE = MeanSquareError()
    assert MSE.objective(y, z) == 0.0

def test_mse_gradient() -> None:
    X = np.array([[0.5, 1], [-1, 1], [7, -6], [6, 8], [1, 1]])
    y = np.array([0.5, 1, -1, 1, -6])
    z = np.array([0, 2, -1, 2, -5])
    MSE = MeanSquareError()
    assert_almost_equal(MSE.gradient(X, y, z), [1.15, 1.9])
    assert MSE.bias_gradient(y, z) == 0.5

def test_mse_gradient_zero() -> None:
    X = np.array([[0.5, 1], [-1, 1], [7, -6], [6, 8], [1, 1]])
    y = np.array([0, 1, 3, 2, 5.78])
    z = np.array([0, 1, 3, 2, 5.78])
    MSE = MeanSquareError()
    assert_almost_equal(MSE.gradient(X, y, z), [0.0, 0.0])
    assert MSE.bias_gradient(y, z) == 0.0
