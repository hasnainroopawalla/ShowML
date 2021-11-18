from showml.deep_learning.network import Sequential
from showml.losses.metrics import accuracy, binary_cross_entropy
from showml.optimizers import RMSProp
from showml.losses import BinaryCrossEntropy
from showml.deep_learning.layers import Dense
from showml.deep_learning.activations import Sigmoid


optimizer = RMSProp(loss_function=BinaryCrossEntropy())

model = Sequential()

model.add(Dense(num_nodes=10, input_shape=(784,)))
model.add(Sigmoid())
model.add(Dense(50))
model.add(Sigmoid())
model.add(Dense(2))
model.add(Sigmoid())

model.compile(optimizer=optimizer, metrics=[binary_cross_entropy, accuracy])
model.summary()
