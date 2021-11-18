import numpy as np
from showml.deep_learning.network import Sequential
from showml.losses.metrics import accuracy, binary_cross_entropy
from showml.optimizers import RMSProp
from showml.losses import BinaryCrossEntropy
from showml.deep_learning.layers import Dense
from showml.deep_learning.activations import Sigmoid, Softmax
from showml.utils.dataset import Dataset
from showml.utils.preprocessing import one_hot_encoding

from sklearn import datasets

data = datasets.load_digits()
X_train = data.data
y_train = one_hot_encoding(data.target)
print(f"X: {X_train.shape}, y: {y_train.shape}")

optimizer = RMSProp(loss_function=BinaryCrossEntropy())

model = Sequential()

model.add(Dense(num_nodes=20, input_shape=(64,)))
model.add(Sigmoid())
model.add(Dense(10))
model.add(Softmax())

model.compile(optimizer=optimizer, metrics=[binary_cross_entropy, accuracy])
model.summary()

model.fit(Dataset(X_train, y_train), batch_size=32, epochs=50)
print(model.predict(X_train)[0])
