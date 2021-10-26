from typing import List
import matplotlib.pyplot as plt
import numpy as np


def plot_loss(losses: List[float]) -> None:
    """
    Plot the loss value at each epoch
    param losses: A list of loss values for each epoch
    """
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Epochs vs Loss")
    plt.show()


def plot_r2_score(r2_scores: List[float]) -> None:
    """
    Plot the r^2 score at each epoch
    param r2_scores: A list of the r^2 score for each epoch
    """
    plt.plot(r2_scores)
    plt.xlabel("Epoch")
    plt.ylabel("R^2 score")
    plt.title("Epochs vs R^2 score")
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
