import numpy as np
from showml.utils.exceptions import DatasetSizeMismatchError, DataTypeError


class Dataset:
    """
    A Dataset class which is instantiated in order to validate the model inputs X and y before training/testing
    param X: The input dataset
    param y: The labels/values of the dataset
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.validate_data_types()
        self.valid_dataset_length()

    def validate_data_types(self) -> None:
        """
        Validates if X and y are NumPy arrays (np.ndarray)
        """
        if not (isinstance(self.X, np.ndarray) and isinstance(self.y, np.ndarray)):
            raise DataTypeError("X and y must be NumPy arrays (np.ndarray)")

    def valid_dataset_length(self) -> None:
        """
        Validates if X and y have the same number of samples
        """
        if self.X.shape[0] != self.y.shape[0]:
            raise DatasetSizeMismatchError(
                "Dimension 1 (number of samples) of X and y must the same"
            )
