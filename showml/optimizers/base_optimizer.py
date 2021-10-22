from typing import Tuple
from showml.optimizers.loss_functions import Loss
from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    def __init__(self, loss_function: Loss, learning_rate: float = 0.005):
        self.learning_rate = learning_rate
        self.loss_function = loss_function

    @abstractmethod
    def calculate_gradient(self, X: np.ndarray, error: np.ndarray) -> Tuple[np.ndarray, np.float64]: pass

    @abstractmethod
    def update_weights(self, X: np.ndarray, y: np.ndarray, z: np.ndarray, weights: np.ndarray, bias: np.float64) -> Tuple[np.ndarray, np.float64]: pass

    @abstractmethod
    def calculate_training_error(self, z: np.ndarray, y: np.ndarray) -> np.ndarray: pass

    @abstractmethod
    def get_cost(self, X: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.float64: pass
