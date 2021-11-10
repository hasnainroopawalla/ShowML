from showml.losses import MeanSquareError
from showml.optimizers import SGD
from showml.supervised.regression import LinearRegression
from showml.utils.dataset import Dataset
from showml.losses.metrics import mean_square_error, r2_score
from showml.utils.plots import plot_regression_line
from showml.utils.preprocessing import normalize
from showml.utils.data_loader import load_salary, load_auto


X_train, y_train = load_auto()
X_train = normalize(X_train)
dataset = Dataset(X_train, y_train)

optimizer = SGD(loss_function=MeanSquareError(), learning_rate=0.001, momentum=0.0)
model = LinearRegression()
model.compile(optimizer=optimizer, metrics=[mean_square_error, r2_score])

model.fit(dataset, epochs=10000)
model.plot_metrics()

plot_regression_line(X_train, y_train, model.predict(X_train))
