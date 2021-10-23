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


def load_salary():
    Auto = (
        pd.read_csv(
            "/Users/hasnain/Projects/ShowML/data/Salary_Data.csv",
            na_values="?",
            dtype={"ID": str},
        )
        .dropna()
        .reset_index()
    )
    X_train = Auto[["YearsExperience"]].values
    # X_train = Auto[['horsepower']].values
    y_train = Auto[["Salary"]].values
    return X_train, y_train


X_train, y_train = load_auto()
X_train = normalize(X_train)
y_train = y_train[:, 0]

optimizer = BatchGradientDescent(loss_function=MeanSquareError(), learning_rate=0.001)
model = LinearRegression(optimizer=optimizer, num_epochs=10000)
model.fit(X_train, y_train)
model.plot_loss()


import matplotlib.pyplot as plt

plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, model.predict(X_train), color="blue")
plt.title("years vs salary")
plt.xlabel("number of years")
plt.ylabel("salary (dollars)")
plt.show()