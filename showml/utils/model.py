from typing import Tuple, Generator
import numpy as np


def initialize_params(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Initialize the weights and bias for the model
    param X: The input training data
    """
    num_samples, num_dimensions = X.shape
    limit = 1 / np.sqrt(num_dimensions)
    weights = np.random.uniform(-limit, limit, (num_dimensions,))
    bias = float()
    return weights, bias


def generate_minibatches(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = False
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate batches of the data based on the batch_size
    This method also handles cases where the number of samples is not divisible by the batch size
    Example:
        X.shape -> (50, 7)
        y.shape -> (50, 1)
        batch_size -> 15
        generate_minibatches(X, y, batch_size) -> batches returned of shapes (for X) -> (15, 7), (15, 7), (15, 7), (5, 7)
    param X: The input data
    param y: The true labels of the data
    param batch_size: An integer which determines the size of each batch (number of samples in each batch)
    param shuffle: A flag which determines if the training set should be shuffled before batches are created
    return X_batch, y_batch: Batches of size batch_size (same np.adarray format as X and y)
    """
    assert X.shape[0] == y.shape[0]
    if shuffle:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, X.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, X.shape[0])
        if shuffle:
            batch = indices[start_idx:end_idx]
        else:
            batch = slice(start_idx, end_idx)
        yield X[batch], y[batch]
