from typing import Tuple
from showml.losses.base_loss import Loss
from showml.utils.metrics import training_error
import numpy as np


class MeanSquareError(Loss):
    def objective(self, y: np.ndarray, z: np.ndarray) -> np.float64:
        return np.average(np.square(training_error(y, z)), axis=0) / 2

    def gradient(
        self, X: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.float64]:
        error = training_error(y, z)
        num_samples = len(error)
        dw = (1 / num_samples) * X.T.dot(error)
        db = (1 / num_samples) * np.sum(error)
        return dw, db


class BinaryCrossEntropy(Loss):
    def objective(self, y: np.ndarray, z: np.ndarray) -> np.float64:
        """
        Also known as Log Loss
        """
        m = y.shape[0]
        epsilon = 1e-20
        z[z == 0] = epsilon
        return -np.sum(y * np.log(z) + (1 - y) * np.log(1 - z)) / m

    def gradient(
        self, X: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.float64]:
        error = training_error(y, z)
        num_samples = len(error)
        dw = (1 / num_samples) * np.dot(X.T, (error))
        db = (1 / num_samples) * np.sum((error))
        return dw, db
