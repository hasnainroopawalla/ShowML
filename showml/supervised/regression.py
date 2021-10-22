from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt


class Regression(ABC):
    def __init__(self, optimizer, num_epochs: int = 1000) -> None:
        """
		Base Regression class
		param learning_rate: the learning rate (how much to update the weights at each iteration)
		param num_epochs: the number of epochs for training
		"""
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.weights = np.array([])
        self.bias = np.float64()
        self.costs: List[float] = []

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Computes a forward pass of the model given the data, weights and bias
        param X: The dataset
        """
        pass

    def plot_cost(self) -> None:
        """
		Plot the cost value at each epoch
		"""
        plt.plot(self.costs)
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.show()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
		This method trains the model given the input X and expected output y
		param X: The input training data
		param y: The ouput of training data
		return: A list of costs at every epoch
		"""
        num_samples, num_dimensions = X.shape

        # Initialize weights
        self.weights = np.ones(num_dimensions)

        for epoch in range(1, self.num_epochs + 1):
            # Forward pass
            z = self.predict(X)
            # Update weights based on the error
            self.weights, self.bias = self.optimizer.update_weights(
                X, y, z, self.weights, self.bias
            )

            z = self.predict(X)
            cost = self.optimizer.get_cost(X, y, z)
            print(cost)
            self.costs.append(cost)


class LinearRegression(Regression):
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias
