from abc import ABC, abstractmethod
from typing import List, Type
import numpy as np
import matplotlib.pyplot as plt
from showml.optimizers.base_optimizer import Optimizer

class Regression(ABC):
    def __init__(self, optimizer: Type[Optimizer], num_epochs: int = 1000) -> None:
        """
		Base Regression class
		param learning_rate: the learning rate (how much to update the weights at each iteration)
		param num_epochs: the number of epochs for training
		"""
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.weights = np.array([])
        self.bias = np.float64()
        self.losses: List[float] = []

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Computes a forward pass of the model given the data, weights and bias
        param X: The dataset
        return: An array of predicted values (forward-pass)
        """
        pass

    def plot_loss(self) -> None:
        """
		Plot the loss value at each epoch
		"""
        plt.plot(self.losses)
        plt.xlabel("Epoch")
        plt.ylabel("loss")
        plt.show()


    def initialize_params(self, X: np.ndarray) -> None:
        """
        Initialize the weights and bias for the model
        param X: The input training data
        """
        num_samples, num_dimensions = X.shape
        self.weights = np.ones(num_dimensions)
        self.bias = np.float64()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
		This method trains the model given the input X and expected output y
		param X: The input training data
		param y: The labels of the training data
		"""
        if not isinstance(X[0][0], np.float64): 
            X = X.astype('float64')
        if not isinstance(y[0], np.float64): 
            y = y.astype('float64')
 
        # X, y = X.astype('float64'), y.astype('float64')
        self.initialize_params(X)

        for epoch in range(1, self.num_epochs + 1):
            # Forward pass
            z = self.predict(X)
            # Update weights based on the error
            self.weights, self.bias = self.optimizer.update_weights(
                X, y, z, self.weights, self.bias
            )

            z = self.predict(X)
            loss = self.optimizer.compute_loss(y, z)
            print(loss)
            self.losses.append(loss)


class LinearRegression(Regression):
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias
