from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np
from showml.optimizers.base_optimizer import Optimizer
from showml.utils.plots import generic_metric_plot
from showml.config.metric_map import metric_map


class Regression(ABC):
    def __init__(
        self, optimizer: Optimizer, num_epochs: int = 1000, classification: bool = False
    ) -> None:
        """
        Base Regression class
        param optimizer: The optimizer to be used for training (showml.optimizers)
        param num_epochs: The number of epochs for training
        param classification: A flag to indicate if the model is dealing with a classification problem or not
        """
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.classification = classification
        self.weights: np.ndarray = np.array([])
        self.bias: np.float64 = np.float64()
        self.history: Dict[str, List[float]] = {}

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Computes a forward pass of the model given the data, current weights and current bias of the model
        param X: The input dataset
        return: An array of predicted values (forward-pass)
        """
        pass

    def evaluate(
        self, epoch: int, X: np.ndarray, y: np.ndarray, metrics: List[str]
    ) -> None:
        """
        Evaluate the model and display all the required metrics (accuracy, r^2 score, etc.)
        param epoch: The current epoch number
        param X: The input dataset
        param y: The true labels of the training data
        param metrics: A list of metrics which have to be calculated and displayed for model evaluation
        """
        z = self.predict(X)

        for metric in metrics:
            if metric not in self.history:
                self.history[metric] = []
            self.history[metric].append(metric_map[metric](y, z))

        display = f"Epoch: {epoch}/{self.num_epochs}"
        for metric in self.history:
            display += f", {metric}: {self.history[metric][-1]}"
        print(display)

    def plot_metrics(self):
        """
        Display the plot after training for the specified metrics
        """
        for metric in self.history:
            generic_metric_plot(metric, self.history[metric])

    def initialize_params(self, X: np.ndarray) -> None:
        """
        Initialize the weights and bias for the model
        param X: The input training data
        """
        num_samples, num_dimensions = X.shape
        self.weights = np.ones(num_dimensions)
        self.bias = np.float64()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        plot: bool = True,
        metrics: List[str] = ["training_error"],
    ) -> None:
        """
        This method trains the model given the input data X and labels y
        param X: The input training data
        param y: The true labels of the training data
        param plot: A flag which determines if the model evaluation plots should be displayed or not
        param metrics: A list of metrics which have to be calculated and displayed for model evaluation
        """
        self.initialize_params(X)

        for epoch in range(1, self.num_epochs + 1):
            # Forward pass
            z = self.predict(X)

            # Update weights based on the error
            self.weights, self.bias = self.optimizer.update_weights(
                X, y, z, self.weights, self.bias
            )

            # Evaluate the model
            self.evaluate(epoch, X, y, metrics)

        if plot:
            self.plot_metrics()


class LinearRegression(Regression):
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias


class LogisticRegression(Regression):
    def sigmoid(self, x) -> np.float64:
        """
        The sigmoid activation function
        param x: The input to the sigmoid function
        return: The output after passing the input through a sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.sigmoid(np.dot(X, self.weights) + self.bias)
