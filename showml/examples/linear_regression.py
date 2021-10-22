
import pandas as pd
from showml.preprocessing.standard import normalize
from showml.optimizers.gradient import BatchGradientDescent
from showml.optimizers.loss_functions import MeanSquareError
from showml.supervised.regression import LinearRegression


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

optimizer = BatchGradientDescent(loss_function=MeanSquareError(), learning_rate=0.005)
model = LinearRegression(optimizer=optimizer, num_epochs=1000)
model.fit(X_train, y_train)


model.plot_cost()
# model.varying_learning_rate_plot(X_train,y_train)
# model.plot_linear_regression_model(X_train,y_train)
