from typing import Tuple
import numpy as np
from showml.deep_learning.base_layer import Layer
from showml.optimizers.base_optimizer import Optimizer


class Dense(Layer):
    """
    A Dense Layer
    """

    def __init__(self, num_nodes, input_shape=None):
        """
        Initializes a Dense layer with the specified number of neurons
        param num_nodes: The number of neurons in the layer
        """
        self.num_nodes = num_nodes
        super().__init__(input_shape=input_shape)

    def initialize_params(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer
        limit = 1 / np.sqrt(self.num_nodes)
        self.weights = np.random.uniform(
            -limit, limit, (self.input_shape[0], self.num_nodes)
        )
        self.bias = np.zeros((1, self.num_nodes))

    def get_params_count(self) -> int:
        return np.prod(self.weights.shape) + np.prod(self.bias.shape)

    def get_output_shape(self) -> Tuple[int]:
        return (self.num_nodes,)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.layer_input = X
        return X.dot(self.weights) + self.bias

    def backward(self, grad: np.ndarray) -> np.ndarray:
        old_weights = self.weights
        dw = self.layer_input.T.dot(grad)
        db = np.sum(grad, axis=0, keepdims=True)

        self.weights, self.bias = self.optimizer.update_weights(
            self.weights, self.bias, dw, db
        )

        return grad.dot(old_weights.T)


class Activation(Layer):
    def __init__(self, input_shape=None):
        super().__init__(input_shape=input_shape, has_weights=False)

    def get_output_shape(self) -> Tuple[int]:
        return self.input_shape

    def get_params_count(self) -> int:
        return 0

    def initialize_params(self, optimizer: Optimizer) -> None:
        pass
