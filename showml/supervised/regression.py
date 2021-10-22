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

    def calculate_error(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Calculate the model error by finding difference between predicted values and true values
        param y: The true values
        param z: The predicted values
        return: Model error
        """
        return z - y

    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> np.float64:
        """
		param X: The input training set
		param y: The expected output of the training set
		return: The cost of the model based on the current weights
		"""
        num_samples = len(X)
        cost = (1 / num_samples) * np.sum((self.predict(X) - y) ** 2)
        return cost

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
            # Compute Error
            error = self.calculate_error(y, z)
            # Update weights based on the error
            self.weights, self.bias = self.optimizer.update_weights(
                X, self.weights, self.bias, error
            )
            print(self.compute_cost(X, y))
            self.costs.append(self.compute_cost(X, y))


class LinearRegression(Regression):
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias


# --- #

import pandas as pd
from showml.preprocessing.standard import normalize
from showml.optimizers.gradient import BatchGradientDescent


def load_auto():
    Auto = (
        pd.read_csv(
            "/Users/hasnain/Projects/ShowML/data/Auto.csv",
            na_values="?",
            dtype={"ID": str},
        )
        .dropna()
        .reset_index()
    )
    X_train = Auto[
        [
            "cylinders",
            "displacement",
            "horsepower",
            "weight",
            "acceleration",
            "year",
            "origin",
        ]
    ].values
    # X_train = Auto[['horsepower']].values
    y_train = Auto[["mpg"]].values
    return X_train, y_train


X_train, y_train = load_auto()

X_train = normalize(X_train)

y_train = y_train[:, 0]

optimizer = BatchGradientDescent(learning_rate=0.005)
model = LinearRegression(optimizer=optimizer, num_epochs=1000)
model.fit(X_train, y_train)


model.plot_cost()
# model.varying_learning_rate_plot(X_train,y_train)
# model.plot_linear_regression_model(X_train,y_train)
