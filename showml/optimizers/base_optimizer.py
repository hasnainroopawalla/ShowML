from typing import Any, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    def __init__(self, learning_rate: float):
        """
        Base Optimzer class
        param loss_function: The loss function to be optimized and for computing the gradient
        param learning_rate: the learning rate (how much to update the weights at each iteration)
        """
        self.learning_rate = learning_rate

    @abstractmethod
    def update_weights(
        self, weights: np.ndarray, bias: Any, dw: np.ndarray, db: float
    ) -> Tuple[np.ndarray, Any]:
        """
        Update the weights of the model using the specified loss function and optimizer
        param weights: The set of training weights of the model
        param bias: The bias value of the model
        param dw: Gradient of the weights w.r.t. loss function
        param db: Gradient of the bias w.r.t. loss function
        return weights, bias: The set of updated weights and bias after optimization for an epoch
        """
        pass
