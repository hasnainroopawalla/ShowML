from showml.losses import BinaryCrossEntropy
from showml.optimizers import SGD
from showml.supervised.regression import LogisticRegression
from showml.utils.dataset import Dataset
from showml.losses.metrics import accuracy, binary_cross_entropy
from showml.utils.data_loader import load_wine


X_train, y_train = load_wine()
dataset = Dataset(X_train, y_train)

model = LogisticRegression()
optimizer = SGD(loss_function=BinaryCrossEntropy(), learning_rate=0.001, momentum=0.8)

model.compile(optimizer=optimizer, metrics=[binary_cross_entropy, accuracy])
model.fit(dataset, batch_size=64, epochs=1000)
model.plot_metrics()
