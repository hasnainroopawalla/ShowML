from abc import ABC, abstractmethod
from typing import Tuple
from showml.optimizers.base_optimizer import Optimizer
import numpy as np


class Layer(ABC):
    """
    A layer class
    """

    def __init__(self, input_shape: Tuple[int] = (0,), has_weights: bool = True):
        """
        param input_shape: The shape of the input (This is set to the previous layer's output shape if it is NOT the first layer of the network)
        param has_weights: A boolean to indicate if this layer has trainable params. Activation layers have this flag set to False
        """
        self.input_shape = input_shape
        self.has_weights = has_weights

    @abstractmethod
    def initialize_params(self, optimizer: Optimizer) -> None:
        """
        Initializes the weights and bias of the layer
        param optimizer: The optimizer to be used for training (showml.optimizers)
        """
        pass

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        A method which computes a forward pass of a layer
        """
        pass

    @abstractmethod
    def backward(self, X: np.ndarray) -> np.ndarray:
        """
        A method which computes a backward pass of a layer
        X here corresponds to the input to the layer in the backward propagation
        """
        pass

    @abstractmethod
    def get_output_shape(self) -> Tuple[int]:
        """
        Returns the output shape of the layer
        """
        pass

    @abstractmethod
    def get_params_count(self) -> int:
        """
        Returns the number of trainable parameters of a layer
        """
        pass