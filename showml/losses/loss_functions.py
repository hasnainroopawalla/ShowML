from typing import Tuple
from showml.losses.base_loss import Loss
import numpy as np


class MeanSquareError(Loss):
    def objective(self, y: np.ndarray, z: np.ndarray) -> float:
        return np.average(np.square(self.training_error(y, z)), axis=0)

    def gradient(
        self, X: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        error = self.training_error(y, z)
        num_samples = len(error)
        dw = (1 / num_samples) * X.T.dot(error)
        db = (1 / num_samples) * np.sum(error)
        return dw, db


class BinaryCrossEntropy(Loss):
    def objective(self, y: np.ndarray, z: np.ndarray) -> float:
        """
        Also known as Log Loss
        """
        num_samples = len(y)
        # Avoid division by zero
        z = np.clip(z, 1e-15, 1 - 1e-15)
        return -(1 / num_samples) * (np.sum(y * np.log(z) + (1 - y) * np.log(1 - z)))

    def gradient(
        self, X: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        error = self.training_error(y, z)
        num_samples = len(error)
        dw = (1 / num_samples) * np.dot(X.T, (error))
        db = (1 / num_samples) * np.sum((error))
        return dw, db
