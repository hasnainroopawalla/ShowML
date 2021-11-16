from showml.utils.dataset import Dataset
import pytest

from showml.utils.exceptions import DataTypeError
import numpy as np

def test_string_input() -> None:
    with pytest.raises(DataTypeError):
        ds = Dataset("X","Y")

def test_list_input() -> None:
    X = [8,307,130,3504,12,70,1]
    y = [15]
    with pytest.raises(DataTypeError):
        ds = Dataset(X,y)
    