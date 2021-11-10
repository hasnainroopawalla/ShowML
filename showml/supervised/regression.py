from abc import ABC, abstractmethod
from typing import Callable, Dict, List

import numpy as np
from showml.optimizers.base_optimizer import Optimizer
from showml.utils.dataset import Dataset
from showml.utils.model_utilities import generate_minibatches, initialize_params
from showml.utils.plots import generic_metric_plot


class Regression(ABC):
    def __init__(self) -> None:
        """
        Base Regression class
        """
        pass

    def compile(self, optimizer: Optimizer, metrics: List[Callable] = []):
        """
        Compiles the model with the specified optimizer and evaluation metrics.
        This method also initializes the model.history object to store metric values during training
        param optimizer: The optimizer to be used for training (showml.optimizers)
        param metrics: A list of metrics which have to be calculated and displayed for model evaluation
        """
        self.optimizer = optimizer
        self.metrics = metrics
        self.history: Dict[str, List[float]] = {
            metric.__name__: [] for metric in self.metrics
        }

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Computes a forward pass of the model given the data, current weights and current bias of the model
        param X: The input dataset
        return: An array of predicted values (forward-pass)
        """
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Evaluate the model and display all the required metrics (accuracy, r^2 score, etc.)
        param X: The input dataset
        param y: The true labels of the training data
        """
        z = self.predict(X)

        for metric in self.metrics:
            self.history[metric.__name__].append(metric(y, z))

        text_to_display = ""
        for metric_name in self.history:
            text_to_display += f", {metric_name}: {self.history[metric_name][-1]}"
        print(text_to_display)

    def plot_metrics(self) -> None:
        """
        Display the plot after training for the specified metrics
        """
        for metric in self.history:
            generic_metric_plot(metric, self.history[metric])

    def fit(self, dataset: Dataset, batch_size: int = 32, epochs: int = 1) -> None:
        """
        This method trains the model given the input data X and labels y
        param dataset: An object of the Dataset class - the input dataset and true labels/values of the dataset
        param batch_size: Number of samples per gradient update
        param epochs: The number of epochs for training
        """
        self.weights, self.bias = initialize_params(dataset.X)

        for epoch in range(1, epochs + 1):
            print(f"Epoch: {epoch}/{epochs}", end="")

            for X_batch, y_batch in generate_minibatches(
                dataset.X, dataset.y, batch_size, shuffle=True
            ):
                # Forward pass
                z = self.predict(X_batch)

                # Update weights based on the error
                self.weights, self.bias = self.optimizer.update_weights(
                    X_batch, y_batch, z, self.weights, self.bias
                )

            # Evaluate the model on the entire dataset
            self.evaluate(dataset.X, dataset.y)


class LinearRegression(Regression):
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias


class LogisticRegression(Regression):
    def sigmoid(self, X) -> np.ndarray:
        """
        The sigmoid activation function
        param X: The input to the sigmoid function
        return: The output after passing the input through a sigmoid function
        """
        return 1 / (1 + np.exp(-X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.sigmoid(np.dot(X, self.weights) + self.bias)
