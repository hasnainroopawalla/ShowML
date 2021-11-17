from typing import Callable, Dict, List
from showml.optimizers.base_optimizer import Optimizer
from showml.deep_learning.layers import Layer


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
        self.layers.append(layer)

    def summary(self) -> None:
        """
        Summarizes the model by displaying all layers and their parameters
        """
        print(f"--- Model: {self.__class__.__name__} ---")
        print("Layers:")
        for layer in self.layers:
            print(f"    {layer.__class__.__name__} - {layer.num_nodes}")
