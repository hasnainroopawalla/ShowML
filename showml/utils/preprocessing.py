import numpy as np


def normalize(X: np.ndarray) -> np.ndarray:
    """Normalize the array along the column by subracting the mean and dividing by the standard deviation.

    Args:
        X (np.ndarray): The array to be normalized.

    Returns:
        np.ndarray: The normalized array.
    """
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def one_hot_encode(X: np.ndarray) -> np.ndarray:
    """This method performs one hot encoding on a NumPy array.

    Args:
        X (np.ndarray): The input array to be encoded.

    Returns:
        np.ndarray: A one hot encoded array.
    """
    one_hot_encoded = np.zeros((X.size, X.max() + 1))
    one_hot_encoded[np.arange(X.size), X] = 1
    return one_hot_encoded
