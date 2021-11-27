from showml.deep_learning.network import Sequential
from showml.losses.loss_functions import BinaryCrossEntropy
from showml.losses.metrics import accuracy, binary_cross_entropy
from showml.optimizers import RMSProp
from showml.losses import CrossEntropy
from showml.deep_learning.layers import Dense
from showml.deep_learning.activations import Relu, Sigmoid, Softmax
from showml.optimizers.optimizer_functions import SGD, AdaGrad
from showml.utils.dataset import Dataset
from showml.utils.preprocessing import one_hot_encode
from showml.utils.data_loader import load_housepricedata, load_wine

from sklearn import datasets

data = datasets.load_digits()
X_train = data.data
y_train = one_hot_encode(data.target)

dataset = Dataset(X_train, y_train)
print(f"X: {dataset.X.shape}, y: {dataset.y.shape}")

model = Sequential()

model.add(Dense(num_nodes=32, input_shape=(64,)))
model.add(Relu())
model.add(Dense(32))
model.add(Relu())
model.add(Dense(10))
model.add(Softmax())

optimizer = SGD()
model.compile(optimizer=optimizer, loss=CrossEntropy(), metrics=[accuracy])

model.summary()

model.fit(dataset, batch_size=32, epochs=1000)
