from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class Loss(ABC):
    @abstractmethod
    def objective(self, y: np.ndarray, z: np.ndarray) -> float:
        """
        The objective cost function is defined here
        param y: The true labels of the training data
        param z: The predicted labels
        return: the loss value based on the loss function
        """
        pass

    @abstractmethod
    def parameter_gradient(
        self, X: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Computes the derivative of the weights and bias w.r.t the objective function
        param X: The training data
        param y: The true labels of the training data
        param z: The predicted labels
        return dw: Gradient of the loss function with respect to the weights
        return db: Gradient of the loss function with respect to the bias
        """
        pass

    @abstractmethod
    def objective_gradient(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the objective function
        param y: The true values
        param z: The predicted values
        return: Gradient of the objective function
        """
        pass

    def training_error(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Calculate the model error by finding difference between predicted values and true values
        param y: The true values
        param z: The predicted values
        return: Training error of the model
        """
        return z - y
