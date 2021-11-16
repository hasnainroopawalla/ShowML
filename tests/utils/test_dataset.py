from typing import List
from showml.utils.dataset import Dataset
import pytest

from showml.utils.exceptions import DataTypeError, DatasetSizeMismatchError
import numpy as np
from numpy.testing import assert_almost_equal


def test_string_input() -> None:
    with pytest.raises(DataTypeError):
        ds = Dataset("X", "Y")


def test_list_input() -> None:
    X = [8, 307, 130, 3504, 12, 70, 1]
    y = [15]
    with pytest.raises(DataTypeError):
        ds = Dataset(X, y)


def test_multi_datatype_list_input() -> None:
    X = ["8", 307.998, "130", 3504.9999, 12, 70, 1]
    y = [15]
    with pytest.raises(DataTypeError):
        ds = Dataset(X, y)


def test_empty_input() -> None:
    X: List[None] = []
    y = [15]
    with pytest.raises(DataTypeError):
        ds = Dataset(X, y)


def test_single_sample_input() -> None:
    X = np.array([[8, 307, 130, 3504, 12, 70, 1]])
    y = np.array([[15]])
    ds = Dataset(X, y)
    assert_almost_equal(ds.X, np.array([[8, 307, 130, 3504, 12, 70, 1]]))
    assert_almost_equal(ds.y, np.array([[15]]))


def test_size_X_mismatch() -> None:
    X = np.array(
        [[8, 454, 220, 4354, 9], [9, 442, 21, 4354, 9], [10, 444, 220, 4354, 10]]
    )
    y = np.array([14])
    with pytest.raises(DatasetSizeMismatchError):
        ds = Dataset(X, y)


def test_size_y_mismatch() -> None:
    X = np.array([[8, 454, 220, 4354, 9]])
    y = np.array([[14], [20], [21]])
    with pytest.raises(DatasetSizeMismatchError):
        ds = Dataset(X, y)


def test_size_mismatch_X_empty() -> None:
    X = np.array([[]])
    y = np.array([[14], [15]])
    with pytest.raises(DatasetSizeMismatchError):
        ds = Dataset(X, y)


def test_size_mismatch_y_empty() -> None:
    X = np.array(
        [[8, 454, 220, 4354, 9], [9, 442, 21, 4354, 9], [10, 444, 220, 4354, 10]]
    )
    y = np.array([[]])
    with pytest.raises(DatasetSizeMismatchError):
        ds = Dataset(X, y)


def test_correct_multisample_input() -> None:
    X = np.array(
        [[8, 454, 220, 4354, 9], [9, 442, 21, 4354, 9], [10, 444, 220, 4354, 10]]
    )
    y = np.array([[14], [20], [21]])
    ds = Dataset(X, y)
    assert_almost_equal(
        ds.X,
        np.array(
            [[8, 454, 220, 4354, 9], [9, 442, 21, 4354, 9], [10, 444, 220, 4354, 10]]
        ),
    )
    assert_almost_equal(ds.y, np.array([[14], [20], [21]]))
