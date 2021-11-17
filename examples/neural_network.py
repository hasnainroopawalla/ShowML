from showml.deep_learning.network import Sequential
from showml.losses.metrics import accuracy, binary_cross_entropy
from showml.optimizers import RMSProp
from showml.losses import BinaryCrossEntropy
from showml.deep_learning.layers import Dense


optimizer = RMSProp(loss_function=BinaryCrossEntropy())

model = Sequential()

model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(20))

model.compile(optimizer=optimizer, metrics=[binary_cross_entropy, accuracy])
model.summary()
