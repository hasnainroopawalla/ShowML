import numpy as np
from showml.losses import MeanSquareError, BinaryCrossEntropy


def r2_score(y: np.ndarray, z: np.ndarray) -> float:
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


def accuracy(y: np.ndarray, z: np.ndarray, logits: bool = True) -> float:
    """
    Compute the classification accuracy of the model
    param y: The true labels
    param z: The predicted labels
    param logits: A flag indicating that the predicted values are probabilites and not classes
    """
    print(y)
    print()
    print(z)
    p = []
    if logits:
        predicted_classes = [1 if i > 0.5 else 0 for i in z]
    return np.sum(y == predicted_classes) / len(y)


def accuracy_2d(y: np.ndarray, z: np.ndarray) -> float:
    """
    Compute the classification accuracy of the model
    param y: The true labels
    param z: The predicted labels
    param logits: A flag indicating that the predicted values are probabilites and not classes
    """
    y = np.argmax(y, axis=1)
    z = np.argmax(z, axis=1)
    return np.sum(y == z) / len(y)


def mean_square_error(y: np.ndarray, z: np.ndarray) -> float:
    return MeanSquareError().objective(y, z)


def binary_cross_entropy(y: np.ndarray, z: np.ndarray) -> float:
    return BinaryCrossEntropy().objective(y, z)
