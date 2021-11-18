from typing import Callable, Dict, List
from showml.optimizers.base_optimizer import Optimizer
from showml.deep_learning.layers import Layer
from terminaltables import AsciiTable


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
