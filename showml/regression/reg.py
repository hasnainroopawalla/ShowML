from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class Regression(ABC):
    def __init__(self, learning_rate: float = 0.005, num_epochs: int = 1000) -> None:
        """
		Base Regression class
		param learning_rate: the learning rate (how much to update the weights at each iteration)
		param num_epochs: the number of epochs for training
		"""
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = []
        self.bias = 0

    @abstractmethod
    def model_forward(self, X, w, b):
        pass

    # @abstractmethod
    # def predict(self, X, w, b):
    # 	pass

    def calculate_error(self, Y, z):
        return z - Y

    def calculate_gradient(self, X, error, num_samples):
        """
		Calculate the gradient of the cost function
		param X: the input training set
		param error: the difference of prediction and actual y values
		param num_samples: the number of input samples
		return: gradient of the cost function (weights and bias)
		"""
        dw = (1 / num_samples) * X.T.dot(error)
        db = (1 / num_samples) * np.sum(error)
        return dw, db

    def update_weights(self, X, error, num_samples):
        """
		Update the weights of the model using Gradient Descent (taking a step in the direction of negative gradient regulated by the learning rate)
		param X: the input training set
		param error: the difference of prediction and actual y values
		return: the updated weights
		"""
        dw, db = self.calculate_gradient(X, error, num_samples)
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def compute_cost(self, X, Y):
        """
		param X: the input training set
		param Y: the expected output of the training set
		return: cost of the model based on the current weights
		"""
        num_samples = len(X)
        cost = (1 / num_samples) * np.sum(
            (self.model_forward(X, self.weights, self.bias) - Y) ** 2
        )
        return cost

    def fit(self, X, Y):
        """
		This method trains the model given the input X and expected output Y
		param X: the input training data
		param Y: the ouput of training data
		return: the training weights, the vector of costs at every epoch
		"""
        cost = []
        num_samples, num_dimensions = X.shape
        self.weights = np.ones(num_dimensions)

        for epoch in range(self.num_epochs):
            z = self.model_forward(X, self.weights, self.bias)
            error = self.calculate_error(Y, z)
            self.update_weights(X, error, num_samples)
            print(self.compute_cost(X, Y))
            cost.append(self.compute_cost(X, Y))
        return cost


class LinearRegression(Regression):
    def model_forward(self, X, w, b):
        return np.dot(X, w) + b

    # def predict(self, X, theta):
    # 	"""
    # 		param X: the input data to be predicted
    # 		param theta: the training weights
    # 		return: prediction of all samples in X
    # 	"""
    # 	X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1) # Append column of 1s for theta0
    # return self.model_forward(X,theta)

    def plot_cost(self, costs):
        """
			param costs: the vector of costs at every epoch
		"""
        plt.plot(costs)
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.show()

    def plot_linear_regression_model(self, X_train, Y_train):
        from showml.preprocessing.standard import normalize

        plt.xlabel("Horsepower")
        plt.ylabel("MPG")
        plt.plot(X_train, Y_train, "o")
        plt.plot(
            X_train, self.model_forward(normalize(X_train), self.weights, self.bias)
        )
        plt.show()


import pandas as pd
from showml.preprocessing.standard import normalize


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
    Y_train = Auto[["mpg"]].values
    return X_train, Y_train


X_train, Y_train = load_auto()
X_train = normalize(X_train)

Y_train = Y_train[:, 0]
model = LinearRegression(num_epochs=1000)
cost = model.fit(X_train, Y_train)

print("Lowest Cost:", cost[-1])
# print('Trained Weights:', theta)

# model.plot_cost(cost)
# model.varying_learning_rate_plot(X_train,Y_train)
# model.plot_linear_regression_model(X_train,Y_train)
