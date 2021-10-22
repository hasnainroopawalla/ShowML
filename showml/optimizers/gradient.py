from typing import Tuple
import numpy as np
from showml.optimizers.base_optimizer import Optimizer


class BatchGradientDescent(Optimizer):
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
        dw, db = self.loss_function.gradient(X, error), self.loss_function.bias_gradient(X, error)
        return dw, db

    def update_weights(
        self, X: np.ndarray, y: np.ndarray, z: np.ndarray, weights: np.ndarray, bias: np.float64
    ) -> Tuple[np.ndarray, np.float64]:
        """
		Update the weights of the model using Gradient Descent (taking a step in the direction of negative gradient regulated by the learning rate)
		param X: The input training set
		param error: The difference between prediction and true values
		"""
        error = self.calculate_training_error(z, y)
        dw, db = self.calculate_gradient(X, error)
        weights -= self.learning_rate * dw
        bias -= self.learning_rate * db
        return weights, bias

    def calculate_training_error(self, z: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate the model error by finding difference between predicted values and true values
        param y: The true values
        param z: The predicted values
        return: Model error
        """
        return z - y

    def get_cost(self, X, y, z):
        error = self.calculate_training_error(z, y)
        return self.loss_function.cost_function(X, y, error)
