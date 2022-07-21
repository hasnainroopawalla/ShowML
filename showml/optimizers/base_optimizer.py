from typing import Tuple
from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    """The Base Optimizer Class."""

    def __init__(self, learning_rate: float):
        """Constructor for the Base Optimzer class.

        Args:
            learning_rate (float): The learning rate (how much to update the weights at each iteration).
        """
        self.learning_rate = learning_rate

    @abstractmethod
    def update_weights(
        self, weights: np.ndarray, bias: np.ndarray, dw: np.ndarray, db: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Update the weights of the model using the specified loss function and optimizer.

        Args:
            weights (np.ndarray): The set of training weights of the model.
            bias (np.ndarray): The bias value(s) of the model.
            dw (np.ndarray): Gradient of the weights w.r.t. loss function.
            db (np.ndarray): Gradient of the bias w.r.t. loss function.

        Returns:
            np.ndarray: The set of updated weights after optimization for an epoch.
            np.ndarray: The set of updated bias value(s) after optimization for an epoch.
        """
        pass
