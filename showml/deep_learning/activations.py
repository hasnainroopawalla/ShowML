from showml.deep_learning.layers import Layer
import numpy as np


class Relu(Layer):
    """
    A layer which applies the ReLU operation to an input
    """

    pass


class Sigmoid(Layer):
    """
    A layer which applies the Sigmoid operation to an input
    """

    def forward(self, X) -> None:
        return 1 / (1 + np.exp(-X))

    def backward(self, X) -> None:
        return self.forward(X) * (1 - self.forward(X))
