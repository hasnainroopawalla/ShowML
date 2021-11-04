from typing import List
import matplotlib.pyplot as plt
import numpy as np


def generic_metric_plot(metric_name: str, metric_values: List[float]) -> None:
    """
    Plot the metric values after training (epoch vs metric)
    param metric_name: Name of the metric (accuracy, r^2 score, etc.)
    param metric_values: A list of metric values collected during training for all epochs
    """
    plt.plot(metric_values)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(metric_name)
    plt.show()


def plot_regression_line(
    X: np.ndarray, y: np.ndarray, pred: np.ndarray, xlabel: str = "", ylabel: str = ""
) -> None:
    """
    Plot the regression line to visualize how well the model fits to the data
    Only works when the entire dataset is 2-dimensional i.e., input data (X) is 1-dimensional
    param X: The input data
    param y: The true labels of the input data
    param pred: The predicted values for the input data
    param xlabel: The label corresponding to the input feature name
    param ylabel: The label corresponding to the output feature name
    """
    assert X.shape[1] == 1
    plt.scatter(X, y, color="red")
    plt.plot(X, pred, color="blue")
    plt.title("Regression Line")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
