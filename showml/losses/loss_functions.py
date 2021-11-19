from typing import Tuple
from showml.losses.base_loss import Loss
import numpy as np


class MeanSquareError(Loss):
    def objective(self, y: np.ndarray, z: np.ndarray) -> float:
        return np.average(np.square(self.training_error(y, z)), axis=0)

    def parameter_gradient(
        self, X: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        error = self.training_error(y, z)
        num_samples = len(error)
        dw = (1 / num_samples) * X.T.dot(error)
        db = (1 / num_samples) * np.sum(error)
        return dw, db


class BinaryCrossEntropy(Loss):
    def objective(self, y: np.ndarray, z: np.ndarray) -> float:
        """Also known as Log Loss"""
        num_samples = len(y)
        # Avoid division by zero
        z = np.clip(z, 1e-15, 1 - 1e-15)
        return -(1 / num_samples) * (np.sum(y * np.log(z) + (1 - y) * np.log(1 - z)))

    def objective_gradient(self, y: np.ndarray, z: np.ndarray) -> float:
        return (z - y) / (z * (1 - z))  # return (-y/z)-(1-y)/(z-1)

    def parameter_gradient(
        self, X: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        error = self.training_error(y, z)
        num_samples = len(error)
        dw = (1 / num_samples) * np.dot(X.T, (error))
        db = (1 / num_samples) * np.sum((error))
        return dw, db


class CrossEntropy(Loss):
    def objective(self, y: np.ndarray, z: np.ndarray) -> float:
        """
        Computes cross entropy between targets (encoded as one-hot vectors)
        and predictions.
        Input: predictions (N, k) ndarray
            targets (N, k) ndarray
        Returns: scalar
        """
        z = np.clip(z, 1e-15, 1 - 1e-15)
        return -y * np.log(z) - (1 - y) * np.log(1 - z)

    def objective_gradient(self, y, z):
        # Avoid division by zero
        z = np.clip(z, 1e-15, 1 - 1e-15)
        return -(y / z) + (1 - y) / (1 - z)

    def parameter_gradient(
        self, X: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        error = self.training_error(y, z)
        num_samples = len(error)
        dw = (1 / num_samples) * np.dot(X.T, (error))
        db = (1 / num_samples) * np.sum((error))
        return dw, db
