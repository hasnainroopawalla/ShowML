from typing import Tuple
import numpy as np


class BatchGradientDescent:
    def __init__(self, learning_rate: float = 0.005):
        self.learning_rate = learning_rate

    def calculate_gradient(
        self, X: np.ndarray, error: np.ndarray
    ) -> Tuple[np.ndarray, np.float64]:
        """
		Calculate the gradient of the cost function
		param X: The input training set
		param error: The difference of prediction and actual y values
		param num_samples: The number of input samples
		return: Gradient of the cost function (weights and bias)
		"""
        num_samples = len(X)
        dw = (1 / num_samples) * X.T.dot(error)
        db = (1 / num_samples) * np.sum(error)
        return dw, db

    def update_weights(
        self, X: np.ndarray, weights: np.ndarray, bias: np.float64, error: np.ndarray
    ) -> Tuple[np.ndarray, np.float64]:
        """
		Update the weights of the model using Gradient Descent (taking a step in the direction of negative gradient regulated by the learning rate)
		param X: The input training set
		param error: The difference between prediction and true values
		"""
        dw, db = self.calculate_gradient(X, error)
        weights -= self.learning_rate * dw
        bias -= self.learning_rate * db
        return weights, bias
