from showml.losses import BinaryCrossEntropy
from showml.optimizers import RMSProp
from showml.supervised.regression import LogisticRegression
from showml.utils.dataset import Dataset
from showml.losses.metrics import accuracy, binary_cross_entropy
from showml.utils.data_loader import load_wine


X_train, y_train = load_wine()
dataset = Dataset(X_train, y_train)

model = LogisticRegression()
optimizer = RMSProp()

model.compile(
    optimizer=optimizer,
    loss=BinaryCrossEntropy(),
    metrics=[binary_cross_entropy, accuracy],
)
model.fit(dataset, batch_size=64, epochs=2000)
model.plot_metrics()
