from showml.losses.loss_functions import MeanSquareError
from showml.optimizers.gradient import BatchGradientDescent
import numpy as np
from numpy.testing import assert_almost_equal

"""
Batch Gradient Descent
"""


def test_update_weights() -> None:
    X = np.array([[0.5, 1], [-1, 1], [7, -6], [6, 8], [1, 1]])
    y = np.array([0.5, 1, -1, 1, -6])
    z = np.array([0, 2, -1, 2, -5])
    weights = np.array([1.4, 7.8])
    bias = 2.6
    optimizer = BatchGradientDescent(
        loss_function=MeanSquareError(), learning_rate=0.01
    )
    updated_weights, updated_bias = optimizer.update_weights(X, y, z, weights, bias)
    assert_almost_equal(updated_weights, [1.3885, 7.781])
    assert updated_bias == 2.595


def test_update_weights_simple() -> None:
    X = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 7.0, 8.0]])
    y = np.array([10.0, 18.0])
    z = np.array([5.0, 7.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    bias = 0.0
    optimizer = BatchGradientDescent(loss_function=MeanSquareError(), learning_rate=0.1)
    updated_weights, updated_bias = optimizer.update_weights(X, y, z, weights, bias)
    assert_almost_equal(updated_weights, [2.35, 2.05, 5.6, 6.4])
    assert updated_bias == 0.8


def test_update_weights_int() -> None:
    X = np.array([[1, 2, 3, 4], [2, 1, 7, 8]]).astype("float64")
    y = np.array([10, 18]).astype("float64")
    z = np.array([5, 7]).astype("float64")
    weights = np.array([1, 1, 1, 1]).astype("float64")
    bias = 0
    optimizer = BatchGradientDescent(loss_function=MeanSquareError(), learning_rate=0.1)
    updated_weights, updated_bias = optimizer.update_weights(X, y, z, weights, bias)
    assert_almost_equal(updated_weights, [2.35, 2.05, 5.6, 6.4])
    assert updated_bias == 0.8


test_update_weights_int()
