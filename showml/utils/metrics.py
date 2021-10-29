import numpy as np
from showml.losses.loss_functions import MeanSquareError, BinaryCrossEntropy


def r2_score(y: np.ndarray, z: np.ndarray) -> np.float64:
    """
    Calculate the r^2 (coefficient of determination) score of the model
    param y: The true values
    param z: The predicted values
    return: The r^2 score
    """
    rss = np.sum(np.square(y - z))
    tss = np.sum(np.square(y - np.mean(y)))
    r_2 = 1 - (rss / tss)
    return r_2


def accuracy(y: np.ndarray, z: np.ndarray, logits: bool = True) -> np.float64:
    """
    Compute the classification accuracy of the model
    param y: The true labels
    param z: The predicted labels
    param logits: A flag indicating that the predicted values are probabilites and not classes
    """
    if logits:
        z = [1 if i > 0.5 else 0 for i in z]
    return np.sum(y == z) / len(y)


def mean_square_error(y: np.ndarray, z: np.ndarray) -> np.float64:
    return MeanSquareError().objective(y, z)


def binary_cross_entropy(y: np.ndarray, z: np.ndarray) -> np.float64:
    return BinaryCrossEntropy().objective(y, z)
