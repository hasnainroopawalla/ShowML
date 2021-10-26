from typing import Tuple
import numpy as np
from showml.optimizers.base_optimizer import Optimizer


class BatchGradientDescent(Optimizer):
    def calculate_gradient(
        self, X: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.float64]:
        """
		Calculate the gradient of the loss function
		param X: The input training set
        param y: The true labels of the training data
        param z: The predicted labels
		return: Gradient of the loss function (weights -> dw and bias -> db)
		"""
        dw, db = (
            self.loss_function.gradient(X, y, z),
            self.loss_function.bias_gradient(y, z),
        )
        return dw, db

    def update_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        weights: np.ndarray,
        bias: np.float64,
    ) -> Tuple[np.ndarray, np.float64]:
        dw, db = self.calculate_gradient(X, y, z)
        weights -= self.learning_rate * dw
        bias -= self.learning_rate * db
        return weights, bias

    def compute_loss(self, y: np.ndarray, z: np.ndarray) -> np.float64:
        return self.loss_function.objective(y, z)
