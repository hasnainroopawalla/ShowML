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
    """Plot the regression line to visualize how well the model fits to the data.
    Only works when the entire dataset is 2-dimensional i.e., input data (X) is 1-dimensional.

    Args:
        y (np.ndarray): The true labels of the input data.
        z (np.ndarray): The predicted values for the input data.
    """
    plt.scatter(y, z, color="red")
    plt.title("Actual vs Predicted")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()
