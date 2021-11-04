from typing import Tuple
from showml.preprocessing.standard import normalize
from showml.optimizers import BatchGradientDescent
from showml.losses import BinaryCrossEntropy
from showml.supervised.regression import LogisticRegression
import numpy as np
import pandas as pd
from showml.utils.metrics import accuracy, binary_cross_entropy


def load_wine() -> Tuple[np.ndarray, np.ndarray]:
    """
    A method to load the Auto.csv file, include the necessary columns and return X_train and y_train
    return X_train: The input training data
    return y_train: The input training labels
    """
    wine = (
        pd.read_csv(
            "./data/Wine_Quality.csv", sep=";", na_values="?", dtype={"ID": str}
        )
        .dropna()
        .reset_index()
    )
    X_train = wine[
        [
            # "fixed acidity",
            # "volatile acidity",
            # "citric acid",
            # "residual sugar",
            # "chlorides",
            # "free sulfur dioxide",
            # "total sulfur dioxide",
            # "density",
            # "pH",
            "sulphates",
            "alcohol",
        ]
    ].values

    # Remap wine quality to only 2 groups
    wine["quality"] = wine["quality"].map({3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 1})
    y_train = wine[["quality"]].values

    # Make y_train 1D if its not
    if y_train.ndim > 1:
        y_train = y_train[:, 0]
    return X_train, y_train


X_train, y_train = load_wine()
# X_train = normalize(X_train)

optimizer = BatchGradientDescent(
    loss_function=BinaryCrossEntropy(), learning_rate=0.001
)
model = LogisticRegression(optimizer=optimizer, num_epochs=100)
model.fit(X_train, y_train, plot=True, metrics=[binary_cross_entropy, accuracy])
