from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    @abstractmethod
    def objective(self, error: np.ndarray) -> np.float64:
        """
        The objective cost function is defined here
        param error: The difference between predicted and true labels
        return the loss value based on the loss function
        """
        pass

    @abstractmethod
    def gradient(self, X: np.ndarray, error: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the objective function (change in weight)
        param X: The training data
        param error: The difference between predicted and true labels
        return dw: change in weight
        """
        pass

    @abstractmethod
    def bias_gradient(self, error: np.ndarray) -> np.float64:
        """
        Computes the gradient of the objective function (change in bias)
        param error: The difference between predicted and true labels
        return db: change in bias
        """
        pass


class MeanSquareError(Loss):
    def objective(self, error: np.ndarray) -> np.float64:
        num_samples = len(error)
        return (1 / (2 * num_samples)) * np.sum(np.square(error))

    def gradient(self, X: np.ndarray, error: np.ndarray) -> np.ndarray:
        num_samples = len(error)
        return (1 / num_samples) * X.T.dot(error)

    def bias_gradient(self, error: np.ndarray) -> np.float64:
        num_samples = len(error)
        return (1 / num_samples) * np.sum(error)
