from typing import Tuple
import numpy as np
from showml.optimizers.base_optimizer import Optimizer


class BatchGradientDescent(Optimizer):
    def calculate_gradient(
        self, X: np.ndarray, error: np.ndarray
    ) -> Tuple[np.ndarray, np.float64]:
        """
		Calculate the gradient of the loss function
		param X: The input training set
		param error: The difference of prediction and actual y values
		return: Gradient of the loss function (weights - dw and bias - db)
		"""
        dw, db = (
            self.loss_function.gradient(X, error),
            self.loss_function.bias_gradient(X, error),
        )
        return dw, db

    def update_weights(
        self, X: np.ndarray, error: np.ndarray, weights: np.ndarray, bias: np.float64
    ) -> Tuple[np.ndarray, np.float64]:
        dw, db = self.calculate_gradient(X, error)
        weights -= self.learning_rate * dw
        bias -= self.learning_rate * db
        return weights, bias

    def get_loss(self, X, y, error):
        return self.loss_function.objective(X, y, error)
