from typing import Tuple
from showml.losses.base_loss import Loss
from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    def __init__(self, loss_function: Loss, learning_rate: float):
        """
        Base Optimzer class
        param loss_function: The loss function to be optimized and for computing the gradient
        param learning_rate: the learning rate (how much to update the weights at each iteration)
        """
        self.loss_function = loss_function
        self.learning_rate = learning_rate

    @abstractmethod
    def update_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        weights: np.ndarray,
        bias: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Update the weights of the model using the specified loss function and optimizer
        param X: The input training set
        param y: The true labels of the training data
        param z: The predicted labels
        param error: The difference between prediction and true values
        param weights: The set of training weights of the model
        param bias: The bias value of the model
        return weights, bias: The set of updated weights and bias after optimization for an epoch
        """
        pass
