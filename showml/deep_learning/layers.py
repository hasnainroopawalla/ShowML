from abc import ABC, abstractmethod

from showml.utils.model_utilities import initialize_params


class Layer(ABC):
    """
    A layer class
    """

    @abstractmethod
    def forward() -> None:
        """
        A method which computes a forward pass of a layer
        """
        pass

    @abstractmethod
    def backward() -> None:
        """
        A method which computes a backward pass of a layer
        """
        pass


class Dense(Layer):
    """
    A Dense Layer
    """

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def initialize_layer(self) -> None:
        self.weights, self.bias = initialize_params(self.num_nodes)
    
    def forward(self, X) -> None:
        pass

    def backward(self, X) -> None:
        pass


class Activation(Layer):
    def __init__(self):
        self.has_weights = False
