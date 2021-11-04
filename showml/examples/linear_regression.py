from typing import Tuple
import numpy as np
import pandas as pd
from showml.preprocessing.standard import normalize
from showml.optimizers import SGD
from showml.losses import MeanSquareError
from showml.supervised.regression import LinearRegression
from showml.utils.plots import plot_regression_line
from showml.utils.metrics import mean_square_error, r2_score


def load_auto() -> Tuple[np.ndarray, np.ndarray]:
    """
    A method to load the Auto.csv file, include the necessary columns and return X_train and y_train
    return X_train: The input training data
    return y_train: The input training labels
    """
    auto = (
        pd.read_csv("./data/Auto.csv", na_values="?", dtype={"ID": str})
        .dropna()
        .reset_index()
    )
    X_train = auto[
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
    y_train = auto[["mpg"]].values

    # Make y_train 1D if its not
    if y_train.ndim > 1:
        y_train = y_train[:, 0]

    return X_train, y_train


def load_salary() -> Tuple[np.ndarray, np.ndarray]:
    """
    A method to load the Salary.csv file, include the necessary columns and return X_train and y_train
    return X_train: The input training data
    return y_train: The input training labels
    """
    salary_data = (
        pd.read_csv("./data/Salary.csv", na_values="?", dtype={"ID": str})
        .dropna()
        .reset_index()
    )
    X_train = salary_data[["YearsExperience"]].values
    y_train = salary_data[["Salary"]].values

    # Make y_train 1D if its not
    if y_train.ndim > 1:
        y_train = y_train[:, 0]

    return X_train, y_train


X_train, y_train = load_auto()
X_train = normalize(X_train)

optimizer = SGD(loss_function=MeanSquareError(), learning_rate=0.001)
model = LinearRegression(optimizer=optimizer, num_epochs=10000)
model.fit(X_train, y_train, plot=True, metrics=[mean_square_error, r2_score])

plot_regression_line(X_train, y_train, model.predict(X_train))
