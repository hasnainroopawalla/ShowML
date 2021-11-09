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
        self.prev_change: Dict[str, Any] = {"weights": np.ndarray([]), "bias": 0.0}
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
