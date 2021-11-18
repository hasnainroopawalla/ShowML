from typing import Callable, Dict, List
import numpy as np
from terminaltables import AsciiTable

from showml.optimizers.base_optimizer import Optimizer
from showml.deep_learning.layers import Layer
from showml.utils.dataset import Dataset
from showml.utils.model_utilities import generate_minibatches


class Sequential:
    """
    A Sequential model (neural network) with various types of layers and activation functions
    """

    def __init__(self) -> None:
        self.layers: List[Layer] = []

    def compile(self, optimizer: Optimizer, metrics: List[Callable] = []) -> None:
        """
        Compiles the model with the specified optimizer and evaluation metrics.
        This method also initializes the model.history object to store metric values during training
        param optimizer: The optimizer to be used for training (showml.optimizers)
        param metrics: A list of metrics which have to be calculated and displayed for model evaluation
        """
        self.optimizer = optimizer
        self.metrics = metrics
        self.history: Dict[str, List[float]] = {
            metric.__name__: [] for metric in self.metrics
        }

    def add(self, layer: Layer) -> None:
        """
        Adds a layer to the network
        """
        if len(self.layers) > 0:
            layer.input_shape = self.layers[-1].get_output_shape()

        if layer.has_weights == True:
            layer.initialize_params()
        self.layers.append(layer)

    def forward_pass(self, X):
        """
        A forward pass of the network
        """
        prev_layer_output = X
        for layer in self.layers:
            prev_layer_output = layer.forward(prev_layer_output)
        return prev_layer_output

    def backward_pass(self, dw, db):
        """
        A backward pass of the network
        """
        for layer in self.layers[::-1]:
            grad = layer.backward(dw, db)
            
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Evaluate the model and display all the required metrics (accuracy, r^2 score, etc.)
        param X: The input dataset
        param y: The true labels of the training data
        """
        z = self.predict(X)

        for metric in self.metrics:
            self.history[metric.__name__].append(metric(y, z))

        text_to_display = ""
        for metric_name in self.history:
            text_to_display += f", {metric_name}: {self.history[metric_name][-1]}"
        print(text_to_display)
        
    def fit(self, dataset: Dataset, batch_size: int = 32, epochs: int = 1) -> None:
        """
        This method trains the model given the input data X and labels y
        param dataset: An object of the Dataset class - the input dataset and true labels/values of the dataset
        param batch_size: Number of samples per gradient update
        param epochs: The number of epochs for training
        """
        for epoch in range(1, epochs + 1):
            print(f"Epoch: {epoch}/{epochs}", end="")

            for X_batch, y_batch in generate_minibatches(
                dataset.X, dataset.y, batch_size, shuffle=True
            ):
                # Forward pass
                z = self.forward_pass(X_batch)
                print(z)
                print(z.shape)
                dw, db = self.optimizer.loss_function.gradient(X_batch, y_batch, z)
                self.backward_pass(dw, db)

            
                # Update weights based on the error
                # self.weights, self.bias = self.optimizer.update_weights(
                #     X_batch, y_batch, z, self.weights, self.bias
                # )

            # Evaluate the model on the entire dataset
            self.evaluate(dataset.X, dataset.y)

    def predict(self, X):
        return self.forward_pass(X)

    def summary(self) -> None:
        """
        Summarizes the model by displaying all layers and their parameters
        """
        total_params = 0
        print(AsciiTable([[self.__class__.__name__]]).table)
        summary_data = [["Layer", "Params", "Output Shape"]]
        for layer in self.layers:
            summary_data.append(
                [
                    layer.__class__.__name__,
                    str(layer.get_params_count()),
                    str(layer.get_output_shape()),
                ]
            )
            total_params += layer.get_params_count()
        table = AsciiTable(summary_data)
        print(table.table)
        print(f"Total Params: {total_params}")
