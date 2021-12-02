from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class Loss(ABC):
    @abstractmethod
    def objective(self, y: np.ndarray, z: np.ndarray) -> float:
        """The objective loss function.

        Args:
            y (np.ndarray): The true labels of the data.
            z (np.ndarray): The predicted labels.

        Returns:
            float: The loss value.
        """
        pass

    @abstractmethod
    def parameter_gradient(
        self, X: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the derivative of the weights and bias w.r.t the objective function.

        Args:
            X (np.ndarray): The training data.
            y (np.ndarray): The true labels of the data.
            z (np.ndarray): The predicted labels.

        Returns:
            np.ndarray: Gradient of the loss function with respect to the weights.
            np.ndarray: Gradient of the loss function with respect to the bias.
        """
        pass

    @abstractmethod
    def objective_gradient(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Computes the gradient of the objective function.

        Args:
            y (np.ndarray): The true values.
            z (np.ndarray): The predicted values.

        Returns:
            np.ndarray: Gradient of the objective function.
        """
        pass
