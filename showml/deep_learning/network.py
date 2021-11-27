from typing import Callable, Dict, List
import numpy as np
from terminaltables import AsciiTable
from showml.losses.base_loss import Loss
from showml.optimizers.base_optimizer import Optimizer
from showml.deep_learning.layers import Layer
from showml.utils.dataset import Dataset
from showml.utils.model_utilities import generate_minibatches
import copy


class Sequential:
    """
    A Sequential model (neural network) with various types of layers and activation functions
    """

    def __init__(self) -> None:
        self.layers: List[Layer] = []

    def compile(
        self, optimizer: Optimizer, loss: Loss, metrics: List[Callable] = []
    ) -> None:
        """
        Compiles the model with the specified optimizer and evaluation metrics.
        This method also initializes the model.history object to store metric values during training
        param optimizer: The optimizer to be used for training (showml.optimizers)
        param metrics: A list of metrics which have to be calculated and displayed for model evaluation
        """
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.history: Dict[str, List[float]] = {
            metric.__name__: [] for metric in self.metrics
        }
        self.initialize_layers()

    def initialize_layers(self) -> None:
        """
        Initializes all the layers with the specified optimizer and parameters
        """
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx > 0:
                # If it is NOT the first layer of the network, input shape = output shape of previous layer
                layer.input_shape = self.layers[layer_idx - 1].get_output_shape()
            if layer.has_weights == True:
                layer.initialize_params(optimizer=copy.deepcopy(self.optimizer))

    def add(self, layer: Layer) -> None:
        """
        Adds a layer to the network
        """
        self.layers.append(layer)

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        """
        Computes a forward pass of the network
        param X: The input to the network
        return: Output of the last layer of the network [shape: (batch_size x num_classes)]
        """
        prev_layer_output = X
        for layer in self.layers:
            prev_layer_output = layer.forward(prev_layer_output)
        return prev_layer_output

    def backward_pass(self, y_batch: np.ndarray, z: np.ndarray) -> None:
        """
        Computes a backward pass of the network (optimize)
        param y_batch: The true labels
        param z: The predicted labels
        """
        # Traverse the layers in the reverse order
        # The gradient of the loss function [shape: (batch_size x num_classes)]
        grad = self.loss.objective_gradient(y_batch, z)
        for layer in self.layers[::-1]:
            grad = layer.backward(grad)

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

    def fit(self, dataset: Dataset, batch_size: int = 32, epochs: int = 50) -> None:
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

                self.backward_pass(y_batch, z)

            # Evaluate the model on the entire dataset
            self.evaluate(dataset.X, dataset.y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Computes a forward pass of the model on the given data
        param X: The input data to the network
        return: Outputs of the last layer of the network [shape: (num_samples_of_X x num_classes)]]
        """
        return self.forward_pass(X)

    def summary(self) -> None:
        """
        Summarizes the model by displaying all layers, their parameters and total number of trainable parameters
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
