from typing import Tuple
import numpy as np
from showml.optimizers.base_optimizer import Optimizer


class SGD(Optimizer):
    def update_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        weights: np.ndarray,
        bias: np.float64,
    ) -> Tuple[np.ndarray, np.float64]:
        dw, db = self.loss_function.gradient(X, y, z)
        weights -= self.learning_rate * dw
        bias -= self.learning_rate * db
        return weights, bias
