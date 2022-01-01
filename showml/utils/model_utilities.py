from typing import Generator, Tuple

import numpy as np

from showml.utils.exceptions import InvalidShapeError


def initialize_params(num_dimensions: int) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize the weights and bias for a model.

    Args:
        num_dimensions (int): The number of dimensions/features needed for initialization of weights and bias.

    Returns:
        np.ndarray: The initialized weights.
        np.ndarray: The initialized bias value.
    """
    limit = 1 / np.sqrt(num_dimensions)
    weights = np.random.uniform(-limit, limit, (num_dimensions,))
    bias = np.zeros(1)
    return weights, bias


def generate_minibatches(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = False
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Generate batches of the data based on the batch_size.
    This method also handles cases where the number of samples is not divisible by the batch size.
    Example:
        X.shape -> (50, 7)
        y.shape -> (50, 1)
        batch_size -> 15
        generate_minibatches(X, y, batch_size) -> batches returned of shapes (for X) -> (15, 7), (15, 7), (15, 7), (5, 7)

    Args:
        X (np.ndarray): The input data.
        y (np.ndarray): The true labels of the data.
        batch_size (int): An integer which determines the size of each batch (number of samples in each batch).
        shuffle (bool, optional): A flag which determines if the training set should be shuffled before batches are created. Defaults to False.

    Yields:
        np.ndarray: Batches for X of size batch_size (same np.adarray format as X and y).
        np.ndarray: Batches for y of size batch_size (same np.adarray format as X and y).
    """
    if X.shape[0] != y.shape[0]:
        raise InvalidShapeError("X and y must have the same number of samples.")

    if shuffle:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, X.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, X.shape[0])
        if shuffle:
            batch = indices[start_idx:end_idx]  # type: ignore
        else:
            batch = slice(start_idx, end_idx)  # type: ignore
        yield X[batch], y[batch]
