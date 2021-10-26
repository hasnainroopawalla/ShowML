from abc import ABC, abstractmethod
from typing import List
import numpy as np
from showml.optimizers.base_optimizer import Optimizer
from showml.utils.metrics import calculate_r2_score
from showml.utils.plots import plot_loss, plot_r2_score


class Regression(ABC):
    def __init__(self, optimizer: Optimizer, num_epochs: int = 1000) -> None:
        """
		Base Regression class
        param optimizer: The optimizer to be used for training (showml.optimizers)
		param num_epochs: The number of epochs for training
		"""
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.weights: np.ndarray = np.array([])
        self.bias: np.float64 = np.float64()
        self.losses: List[float] = []
        self.r2_scores: List[float] = []

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Computes a forward pass of the model given the data, current weights and current bias of the model
        param X: The input dataset
        return: An array of predicted values (forward-pass)
        """
        pass

    def initialize_params(self, X: np.ndarray) -> None:
        """
        Initialize the weights and bias for the model
        param X: The input training data
        """
        num_samples, num_dimensions = X.shape
        self.weights = np.ones(num_dimensions)
        self.bias = np.float64()

    def fit(self, X: np.ndarray, y: np.ndarray, plot: bool = True) -> None:
        """
		This method trains the model given the input data X and labels y
		param X: The input training data
		param y: The true labels of the training data
        param plot: A flag which determines if the model evaluation plots should be displayed or not
		"""
        self.initialize_params(X)

        for epoch in range(1, self.num_epochs + 1):
            # Forward pass
            z = self.predict(X)

            # Update weights based on the error
            self.weights, self.bias = self.optimizer.update_weights(
                X, y, z, self.weights, self.bias
            )

            # Compute loss on the entire training set
            z = self.predict(X)
            loss = self.optimizer.compute_loss(y, z)
            r2_score = calculate_r2_score(y, z)

            print(
                f"Epoch: {epoch}/{self.num_epochs}, Loss: {loss}, R^2 score: {r2_score}"
            )

            self.losses.append(loss)
            self.r2_scores.append(r2_score)

        if plot:
            plot_loss(self.losses)
            plot_r2_score(self.r2_scores)


class LinearRegression(Regression):
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias
