from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    @abstractmethod
    def objective(self, y: np.ndarray, z: np.ndarray) -> np.float64:
        """
        The objective cost function is defined here
        param y: The true labels of the training data
        param z: The predicted labels
        return: the loss value based on the loss function
        """
        pass

    @abstractmethod
    def gradient(self, X: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the objective function (change in weight)
        param X: The training data
        param y: The true labels of the training data
        param z: The predicted labels
        return dw: change in weight
        """
        pass

    @abstractmethod
    def bias_gradient(self, y: np.ndarray, z: np.ndarray) -> np.float64:
        """
        Computes the gradient of the objective function (change in bias)
        param y: The true labels of the training data
        param z: The predicted labels
        return db: change in bias
        """
        pass
