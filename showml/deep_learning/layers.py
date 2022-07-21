from typing import Tuple
import numpy as np
from showml.deep_learning.base_layer import Layer
from showml.optimizers.base_optimizer import Optimizer


class Dense(Layer):
    """A Dense Layer."""

    def __init__(self, num_nodes: int, input_shape: Tuple[int] = (0,)):
        """Constructor for the Dense Layer.

        Args:
            num_nodes (int): Initializes a Dense layer with the specified number of neurons.
            input_shape (Tuple[int], optional): The number of neurons in the layer. Defaults to (0,).
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
        return int(np.prod(self.weights.shape) + np.prod(self.bias.shape))

    def get_output_shape(self) -> Tuple[int]:
        return (self.num_nodes,)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.layer_input = X
        return X.dot(self.weights) + self.bias

    def backward(self, X: np.ndarray) -> np.ndarray:
        old_weights = self.weights

        dw = self.layer_input.T.dot(X)
        db = np.sum(X, axis=0, keepdims=True)

        self.weights, self.bias = self.optimizer.update_weights(
            self.weights, self.bias, dw, db
        )

        return X.dot(old_weights.T)


class Activation(Layer):
    def __init__(self, input_shape: Tuple[int] = (0,)):
        super().__init__(input_shape=input_shape, has_weights=False)

    def get_output_shape(self) -> Tuple[int]:
        return self.input_shape

    def get_params_count(self) -> int:
        return 0

    def initialize_params(self, optimizer: Optimizer) -> None:
        pass
