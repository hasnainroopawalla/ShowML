from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
    @abstractmethod
    def cost_function(self, X: np.ndarray, y: np.ndarray, error: np.ndarray) -> np.float64: pass

    @abstractmethod
    def gradient(self, X: np.ndarray, error: np.ndarray) -> np.ndarray: pass

    @abstractmethod
    def bias_gradient(self, X: np.ndarray, error: np.ndarray) -> np.float64: pass


class MeanSquareError(Loss):
    def cost_function(self, X: np.ndarray, y: np.ndarray, error: np.ndarray) -> np.float64:
        num_samples = len(X)
        cost = (1 / num_samples) * np.sum((error) ** 2)
        return cost

    def gradient(self, X: np.ndarray, error: np.ndarray) -> np.ndarray:
        num_samples = len(X)
        return (1 / num_samples) * X.T.dot(error)

    def bias_gradient(self, X: np.ndarray, error: np.ndarray) -> np.float64:
        num_samples = len(X)
        return (1 / num_samples) * np.sum(error)
