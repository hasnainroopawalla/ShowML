import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class LinearRegressionWithGradientDescent():
	def initialize_parameters(self, X):
		'''
			param X: the input training set
			return: the input training set prepended with a column of 1s, a numpy 1D array with same number of 0s as the input dimensions
		'''
		num_samples, num_dimensions = X.shape[0], X.shape[1]
		X = np.concatenate((np.ones((num_samples,1)),X),axis=1) # Append column of 1s for theta0
		theta = np.ones(num_dimensions+1)
		return X, theta

	def model_forward(self, X, theta):
		'''
			param X: the input data to be predicted
			param theta: the training weights
			return: prediction based on the weights 
		'''
		prediction = X.dot(theta)
		return prediction

	def compute_cost(self, X, Y, theta):
		'''
			param X: the input training set
			param Y: the output of the training set
			param theta: the training weights
			return: cost of the model based on the current weights
		'''
		num_samples = len(X)
		cost = (1/num_samples)*np.sum((X.dot(theta)-Y)**2)
		return cost

	def model_backward(self, X, error, num_samples):
		'''
			param X: the input training set
			param error: the difference of prediction and actual y values
			param num_samples: the number of input samples
			return: gradient of the cost function
		'''
		gradient = (-2/num_samples)*X.T.dot(error)
		return gradient

	def update_parameters(self, theta, X, error, lr, num_samples):
		'''
			param theta: the training weights
			param X: the input training set
			param error: the difference of prediction and actual y values
			param lr: the learning rate
			param num_samples: the number of input samples
			return: the new values of the weights after taking a step in the direction of negative gradient regulated by the learning rate
		'''
		gradient = self.model_backward(X,error,num_samples)
		theta = theta - lr * gradient
		return theta

	def predict(self, X, theta):
		'''
			param X: the input data to be predicted
			param theta: the training weights
			return: prediction of all samples in X
		'''
		X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1) # Append column of 1s for theta0
		return self.model_forward(X,theta)

	def normalize(self, arr):
		'''
			param arr: the matrix to be normalized
			return: the normalized matrix
		'''
		arr = (arr-np.mean(arr, axis=0))/np.std(arr, axis=0)
		return arr

	def plot_cost(self, costs):
		'''
			param costs: the vector of costs at every epoch
		'''
		plt.plot(costs)
		plt.xlabel('Epoch')
		plt.ylabel('Cost')
		plt.show()

	def train_linear_model(self, X, Y, lr=0.005, epochs=1000, normalize=True):
		'''
			param X: the input training data
			param Y: the ouput of training data
			param lr: the learning rate
			param epochs: the number of iterations for training
			return: the training weights, the vector of costs at every epoch
		'''
		X = self.normalize(X) if normalize == True else X
		cost = []
		num_samples, num_dimensions = X.shape[0], X.shape[1]
		'''
			X:
			[12 4],
			[23 5],
			[33 1],
			[52 6]

			num_samples = 4
			num_dimensions = 3
		'''

		X, theta = self.initialize_parameters(X)
		'''
			X:
				[[1 12 4],
				[1 23 5],
				[1 33 1],
				[1 52 6]]
			theta:
				[0,
				0,
				0]
		'''
		Y = Y[:,0] # To make Y 1-dimensional
		
		for i in range(epochs):
			z = self.model_forward(X,theta)
			error = Y - z
			theta = self.update_parameters(theta,X,error,lr,num_samples)
			cost.append(self.compute_cost(X,Y,theta))
		return theta, cost

	
	# def plot_linear_regression_model(self,X_train,Y_train):
	# 	plt.xlabel('Horsepower')
	# 	plt.ylabel('MPG')
	# 	plt.plot(X_train, Y_train, 'o')
	# 	plt.plot(X_train, self.predict(self.normalize(X_train),theta))
	# 	plt.show()
