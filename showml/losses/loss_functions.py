from typing import Tuple
from showml.losses.base_loss import Loss
import numpy as np


class MeanSquaredError(Loss):
    def objective(self, y: np.ndarray, z: np.ndarray) -> float:
        return np.average(np.square(z - y), axis=0)

    def objective_gradient(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        # TODO
        pass

    def parameter_gradient(
        self, X: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        error = z - y
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

    def objective_gradient(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, 1e-15, 1 - 1e-15)
        return (-y / z) - (1 - y) / (z - 1)

    def parameter_gradient(
        self, X: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        error = z - y
        num_samples = len(error)
        dw = (1 / num_samples) * np.dot(X.T, (error))
        db = (1 / num_samples) * np.sum((error))
        return dw, db


class CrossEntropy(Loss):
    def objective(self, y: np.ndarray, z: np.ndarray) -> float:
        """param y: one hot encoded values"""
        num_samples = len(z)
        z = np.clip(z, 1e-15, 1.0 - 1e-15)
        return -np.sum(y * np.log(z)) / num_samples

    def objective_gradient(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, 1e-15, 1 - 1e-15)
        return -(y / z) + (1 - y) / (1 - z)

    def parameter_gradient(
        self, X: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # TODO
        pass
