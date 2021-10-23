from typing import Tuple
from showml.optimizers.loss_functions import Loss
from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    def __init__(self, loss_function: Loss, learning_rate: float = 0.005):
        self.learning_rate = learning_rate
        self.loss_function = loss_function

    @abstractmethod
    def update_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        weights: np.ndarray,
        bias: np.float64,
    ) -> Tuple[np.ndarray, np.float64]:
        """
		Update the weights of the model using the specified loss function and optimizer
		param X: The input training set
        param error: The difference between prediction and true values
		param weights: The set of training weights
        param bias: The bias value
        return weights, bias: The set of updated weights and bias after optimization
		"""
        pass

    @abstractmethod
    def compute_loss(self, y: np.ndarray, z: np.ndarray) -> np.float64:
        """
        Compute the loss of the model based on the specified loss function
        param X: The input training set
        param y: The labels of the training data
        param z: The predicted labels
        return: The loss value of the model
        """
        pass

    def calculate_training_error(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Calculate the model error by finding difference between predicted values and true values
        param z: The predicted values
        param y: The true values
        return: Training error of the model
        """
        return z - y
