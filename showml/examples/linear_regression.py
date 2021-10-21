import pandas as pd
from showml.regression.reg import LinearRegression
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


def main():
    X_train, Y_train = load_auto()
    X_train = normalize(X_train)

    model = LinearRegression()
    cost = model.fit(X_train, Y_train)

    print("Lowest Cost:", cost[-1])
    # print('Trained Weights:', theta)

    model.plot_cost(cost)
    # model.varying_learning_rate_plot(X_train,Y_train)
    # model.plot_linear_regression_model(X_train,Y_train)


main()
