from typing import List
import matplotlib.pyplot as plt
import numpy as np


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


def plot_actual_vs_predicted(y: np.ndarray, z: np.ndarray) -> None:
    """Generates a scatter plot of the true values and predicted values.
    A diagonal line from (0, 0) to (+limit, +limit) indicates a very good fit i.e., the true values and predicted values are almost equal.

    Args:
        y (np.ndarray): The true labels of the input data.
        z (np.ndarray): The predicted values for the input data.
    """
    plt.scatter(y, z, color="red")
    plt.title("Actual vs Predicted")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()
