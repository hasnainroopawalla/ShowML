from showml.losses.base_loss import Loss 
import numpy as np


class MeanSquareError(Loss):
    def objective(self, y: np.ndarray, z: np.ndarray) -> np.float64:
        return np.average(np.square(self.calculate_training_error(y, z)), axis=0) / 2

    def gradient(self, X: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        error = self.calculate_training_error(y, z)
        num_samples = len(error)
        return (1 / num_samples) * X.T.dot(error)

    def bias_gradient(self, y: np.ndarray, z: np.ndarray) -> np.float64:
        error = self.calculate_training_error(y, z)
        num_samples = len(error)
        return (1 / num_samples) * np.sum(error)
