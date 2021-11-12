from typing import Any, Dict, Tuple
import numpy as np
from showml.optimizers.base_optimizer import Optimizer
from showml.losses.base_loss import Loss


class SGD(Optimizer):
    def __init__(
        self, loss_function: Loss, learning_rate: float = 0.001, momentum: float = 0.0
    ):
        """
        The Stochastic Gradient Descent Optimizer
        param momentum: This parameter builds inertia to overcome noisy gradients. Previous gradient values are stored and used to update the weights and bias at the current step
        """
        assert 0.0 <= momentum <= 1.0
        self.momentum = momentum
        self.prev_change: Dict[str, Any] = {"weights": np.array([]), "bias": 0.0}
        super().__init__(loss_function, learning_rate)

    def update_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        weights: np.ndarray,
        bias: float,
    ) -> Tuple[np.ndarray, float]:
        dw, db = self.loss_function.gradient(X, y, z)

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
    def __init__(
        self, loss_function: Loss, learning_rate: float = 0.01, epsilon: float = 1e-8
    ):
        """
        The Adaptive Gradient Descent Optimizer
        Reference: https://ruder.io/optimizing-gradient-descent/index.html#adagrad
        param epsilon: A smoothing term that ensures the denominator is > 0 (to avoid division by zero)
        """
        self.G: Dict[str, Any] = {"weights": np.array([]), "bias": 0.0}
        self.epsilon = epsilon
        super().__init__(loss_function, learning_rate)

    def update_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        weights: np.ndarray,
        bias: float,
    ) -> Tuple[np.ndarray, float]:
        dw, db = self.loss_function.gradient(X, y, z)

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
        self,
        loss_function: Loss,
        learning_rate: float = 0.001,
        rho: float = 0.9,
        epsilon: float = 1e-8,
    ):
        """
        The Root Mean Squared Propagation (RMSProp) Optimizer
        Reference: https://machinelearningmastery.com/gradient-descent-with-rmsprop-from-scratch/
        param rho: A parameter to determine the weight to be given to recent gradients as compared to older gradients
        """
        self.G: Dict[str, Any] = {"weights": np.array([]), "bias": 0.0}
        self.rho = rho
        self.epsilon = epsilon
        super().__init__(loss_function, learning_rate)

    def update_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        weights: np.ndarray,
        bias: float,
    ) -> Tuple[np.ndarray, float]:
        dw, db = self.loss_function.gradient(X, y, z)

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
