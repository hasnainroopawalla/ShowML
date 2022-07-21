from showml.deep_learning.layers import Activation
import numpy as np


class Sigmoid(Activation):
    """A layer which applies the Sigmoid operation to an input."""

    def forward(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-X))

    def backward(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X) * (1 - self.forward(X))


class Relu(Activation):
    """A layer which applies the ReLU operation to an input."""

    def forward(self, X: np.ndarray) -> np.ndarray:
        return abs(X) * (X > 0)

    def backward(self, X: np.ndarray) -> np.ndarray:
        return 1.0 * (X > 0)


class Softmax(Activation):
    """A layer which applies the Softmax operation to an input."""

    def forward(self, X: np.ndarray) -> np.ndarray:
        e_x = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def backward(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X) * (1 - self.forward(X))
