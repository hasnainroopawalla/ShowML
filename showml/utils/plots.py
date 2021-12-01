from typing import List
import matplotlib.pyplot as plt
import numpy as np

from showml.utils.exceptions import InvalidShapeError


def generic_metric_plot(metric_name: str, metric_values: List[float]) -> None:
    """Plot the metric values after training (epoch vs metric).

    Args:
        metric_name (str): Name of the metric (accuracy, r^2 score, etc.).
        metric_values (List[float]): A list of metric values collected during training for all epochs.
    """
    plt.plot(metric_values)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(metric_name)
    plt.show()


def plot_regression_line(
    X: np.ndarray, y: np.ndarray, z: np.ndarray, xlabel: str = "", ylabel: str = ""
) -> None:
    """Plot the regression line to visualize how well the model fits to the data.
    Only works when the entire dataset is 2-dimensional i.e., input data (X) is 1-dimensional.

    Args:
        X (np.ndarray): The input data.
        y (np.ndarray): The true labels of the input data.
        z (np.ndarray): The predicted values for the input data.
        xlabel (str, optional): The label corresponding to the input feature name. Defaults to "".
        ylabel (str, optional): The label corresponding to the output feature name. Defaults to "".
    """
    if X.shape[1] != 1:
        raise InvalidShapeError("X must have exactly 1 dimension.")

    plt.scatter(X, y, color="red")
    plt.plot(X, z, color="blue")
    plt.title("Regression Line")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
