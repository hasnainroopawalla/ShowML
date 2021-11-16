from showml.losses import BinaryCrossEntropy
from showml.optimizers.optimizer_functions import RMSProp
from showml.supervised.regression import LogisticRegression
from showml.utils.dataset import Dataset
from showml.losses.metrics import accuracy, binary_cross_entropy
from showml.utils.data_loader import load_wine


X_train, y_train = load_wine()
dataset = Dataset(X_train, y_train)

model = LogisticRegression()
optimizer = RMSProp(loss_function=BinaryCrossEntropy())

model.compile(optimizer=optimizer, metrics=[binary_cross_entropy, accuracy])
model.fit(dataset, batch_size=64, epochs=2000)
model.plot_metrics()
