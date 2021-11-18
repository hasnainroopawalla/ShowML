import numpy as np


def normalize(matrix: np.ndarray) -> np.ndarray:
    """
    Normalize the matrix along the column by subracting the mean and dividing by the standard deviation
    param arr: The matrix to be normalized
    return: The normalized matrix
    """
    return (matrix - np.mean(matrix, axis=0)) / np.std(matrix, axis=0)


def one_hot_encoding(X: np.ndarray) -> np.ndarray:
    b = np.zeros((X.size, X.max() + 1))
    b[np.arange(X.size), X] = 1
    return b
