import numpy as np


def training_error(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Calculate the model error by finding difference between predicted values and true values
    param y: The true values
    param z: The predicted values
    return: Training error of the model
    """
    return z - y


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


def accuracy(y: np.ndarray, z: np.ndarray) -> np.float64:
    return np.sum(y == z) / len(y)
