import numpy as np


def normalize(matrix: np.array) -> np.array:
    """
    Normalize the matrix along the column by subracting the mean and dividing by the standard deviation
    param arr: The matrix to be normalized
    return: The normalized matrix
    """
    return (matrix - np.mean(matrix, axis=0)) / np.std(matrix, axis=0)
