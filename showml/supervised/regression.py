from abc import ABC, abstractmethod
from typing import List
import numpy as np
from showml.optimizers.base_optimizer import Optimizer
from showml.utils.metrics import r2_score, accuracy
from showml.utils.plots import plot_loss, plot_r2_score, plot_accuracy


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
        self.losses: List[float] = []
        self.r2_scores: List[float] = []
        self.accuracies: List[float] = []

    @abstractmethod
    def model_forward(self, X: np.ndarray) -> np.ndarray:
        """
        Computes a forward pass of the model given the data, current weights and current bias of the model
        param X: The input dataset
        return: An array of predicted values (forward-pass)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the values/classes, given the data, current weights and current bias of the model
        param X: The input dataset
        return: An array of predicted values
        """
        pass

    def evaluate(self, epoch: int, X: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
        loss = self.optimizer.compute_loss(y, z)
        r2 = r2_score(y, z)
        acc = None
        if self.classification:
            acc = accuracy(y, self.predict(X))
            self.accuracies.append(acc)

        print(
            f"Epoch: {epoch}/{self.num_epochs}, Acc: {acc}, Loss: {loss}, R^2 score: {r2}"
        )

        self.losses.append(loss)
        self.r2_scores.append(r2)

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
            z = self.model_forward(X)

            # Update weights based on the error
            self.weights, self.bias = self.optimizer.update_weights(
                X, y, z, self.weights, self.bias
            )

            # Compute loss on the entire training set
            z = self.model_forward(X)

            self.evaluate(epoch, X, y, z)

        if plot:
            plot_loss(self.losses)
            plot_r2_score(self.r2_scores)
            if self.classification:
                plot_accuracy(self.accuracies)


class LinearRegression(Regression):
    def model_forward(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias


class LogisticRegression(Regression):
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def model_forward(self, X: np.ndarray):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    def predict(self, X: np.ndarray):
        z = self.sigmoid(np.dot(X, self.weights) + self.bias)
        return [1 if i > 0.5 else 0 for i in z]
