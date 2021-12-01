from typing import Any, Dict, Tuple

import numpy as np
from showml.optimizers.base_optimizer import Optimizer
from showml.utils.exceptions import InvalidValueError


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.001, momentum: float = 0.0):
        """The Stochastic Gradient Descent Optimizer (SGD).

        Args:
            momentum (float, optional): This parameter builds inertia to overcome noisy gradients. Previous gradient values are stored and used to update the weights and bias at the current step. Defaults to 0.0.
        """
        if momentum < 0.0 or momentum > 1.0:
            raise InvalidValueError("Momentum value must be between 0-1")

        self.momentum = momentum
        self.prev_change: Dict[str, Any] = {"weights": np.array([]), "bias": 0.0}
        super().__init__(learning_rate)

    def update_weights(
        self, weights: np.ndarray, bias: np.ndarray, dw: np.ndarray, db: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.prev_change["weights"].shape[0] == 0:
            self.prev_change["weights"] = np.zeros(np.shape(weights))
            self.prev_change["bias"] = np.zeros(np.shape(bias))

        curr_change_weights = self.learning_rate * dw + (
            self.momentum * self.prev_change["weights"]
        )
        curr_change_bias = self.learning_rate * db + (
            self.momentum * self.prev_change["bias"]
        )

        weights -= curr_change_weights
        bias -= curr_change_bias

        # Store the current gradient for the next iteration
        self.prev_change["weights"] = curr_change_weights
        self.prev_change["bias"] = curr_change_bias

        return weights, bias


class AdaGrad(Optimizer):
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        """The Adaptive Gradient Descent Optimizer.
        Reference: https://ruder.io/optimizing-gradient-descent/index.html#adagrad

        Args:
            epsilon (float, optional): A smoothing term that ensures the denominator is > 0 (to avoid division by zero). Defaults to 1e-8.
        """
        self.G: Dict[str, Any] = {"weights": np.array([]), "bias": 0.0}
        self.epsilon = epsilon
        super().__init__(learning_rate)

    def update_weights(
        self, weights: np.ndarray, bias: np.ndarray, dw: np.ndarray, db: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        if self.G["weights"].shape[0] == 0:
            self.G["weights"] = np.zeros(np.shape(weights))
            self.G["bias"] = np.zeros(np.shape(bias))

        self.G["weights"] += np.square(dw)
        self.G["bias"] += np.square(db)

        weights -= self.learning_rate * (dw / np.sqrt(self.G["weights"] + self.epsilon))
        bias -= self.learning_rate * (db / np.sqrt(self.G["bias"] + self.epsilon))

        return weights, bias


class RMSProp(Optimizer):
    def __init__(
        self, learning_rate: float = 0.001, rho: float = 0.9, epsilon: float = 1e-8
    ):
        """The Root Mean Squared Propagation (RMSProp) Optimizer.
        Reference: https://machinelearningmastery.com/gradient-descent-with-rmsprop-from-scratch/

        Args:
            rho (float, optional): A parameter to determine the weight to be given to recent gradients as compared to older gradients. Defaults to 0.9.
            epsilon (float, optional): A smoothing term that ensures the denominator is > 0 (to avoid division by zero). Defaults to 1e-8.
        """
        self.G: Dict[str, Any] = {"weights": np.array([]), "bias": 0.0}
        self.rho = rho
        self.epsilon = epsilon
        super().__init__(learning_rate)

    def update_weights(
        self, weights: np.ndarray, bias: np.ndarray, dw: np.ndarray, db: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        if self.G["weights"].shape[0] == 0:
            self.G["weights"] = np.zeros(np.shape(weights))
            self.G["bias"] = np.zeros(np.shape(bias))

        self.G["weights"] = (self.rho * self.G["weights"]) + (
            (1 - self.rho) + np.square(dw)
        )
        self.G["bias"] = (self.rho * self.G["bias"]) + ((1 - self.rho) + np.square(db))

        weights -= self.learning_rate * (dw / np.sqrt(self.G["weights"] + self.epsilon))
        bias -= self.learning_rate * (db / np.sqrt(self.G["bias"] + self.epsilon))

        return weights, bias
