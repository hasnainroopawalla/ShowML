import pandas as pd
from showml.regression.linear_regression import LinearRegressionWithGradientDescent

def load_auto():
	Auto = pd.read_csv('/Users/hasnain/Projects/ShowML/data/Auto.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()
	X_train = Auto[['cylinders','displacement','horsepower','weight', 'acceleration','year','origin']].values
	# X_train = Auto[['horsepower']].values
	Y_train = Auto[['mpg']].values
	return X_train, Y_train

X_train, Y_train = load_auto()

model = LinearRegressionWithGradientDescent()
theta, cost = model.train_linear_model(X_train,Y_train,lr=0.01,epochs=1000)

print('Lowest Cost:', cost[-1])
print('Trained Weights:', theta)

model.plot_cost(cost)
# model.varying_learning_rate_plot(X_train,Y_train)
# model.plot_linear_regression_model(X_train,Y_train)