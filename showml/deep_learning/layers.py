from abc import ABC, abstractmethod

import numpy as np

from showml.utils.model_utilities import initialize_params


class Layer(ABC):
    """
    A layer class
    """

    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        self.has_weights = True

    @abstractmethod
    def initialize_params(self) -> None:
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
        """
        pass

    @abstractmethod
    def get_output_shape(self) -> None:
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


class Dense(Layer):
    """
    A Dense Layer
    """

    def __init__(self, num_nodes, input_shape=None):
        """
        param num_nodes: The number of neurons in the layer
        param input_shape: A tuple indicating the shape of the input to the layer (to be specified if is the first layer of the network)
        """
        self.num_nodes = num_nodes
        self.input_shape = input_shape
        self.has_weights = True

    def initialize_params(self) -> None:
        """
        Initializes the weights and bias of the Dense layer
        """
        limit = 1 / np.sqrt(self.num_nodes)
        self.weights = np.random.uniform(
            -limit, limit, (self.input_shape[0], self.num_nodes)
        )
        self.bias = np.zeros((1, self.num_nodes))

    def get_params_count(self) -> int:
        """
        Computes and returns the total number of trainable parameters for the layer
        """
        return np.prod(self.weights.shape) + np.prod(self.bias.shape)

    def get_output_shape(self):
        return (self.num_nodes,)

    def forward(self, X) -> np.ndarray:
        pass

    def backward(self, X) -> np.ndarray:
        pass


class Activation(Layer):
    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        self.has_weights = False

    def get_output_shape(self) -> None:
        return self.input_shape

    def get_params_count(self) -> int:
        return 0

    def initialize_params(self) -> None:
        pass
