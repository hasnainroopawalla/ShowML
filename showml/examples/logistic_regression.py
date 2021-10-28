from showml.preprocessing.standard import normalize
from showml.optimizers.gradient_optimizers import BatchGradientDescent
from showml.losses.loss_functions import BinaryCrossEntropy
from showml.supervised.regression import LogisticRegression


from sklearn.datasets import make_classification

X_train, y_train = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
)


optimizer = BatchGradientDescent(
    loss_function=BinaryCrossEntropy(), learning_rate=0.001
)
model = LogisticRegression(optimizer=optimizer, num_epochs=1000, classification=True)
model.fit(X_train, y_train, plot=True)
