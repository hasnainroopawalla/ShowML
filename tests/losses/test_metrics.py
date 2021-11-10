from showml.losses.metrics import r2_score
import numpy as np


def test_r2_score_simple() -> None:
    y = np.array([4, 3, 2, 1])
    z = np.array([1, 2, 3, 4])
    assert r2_score(y, z) == -3.0


def test_r2_score_float_1D() -> None:
    y = np.array([3, -0.5, 2, 7])
    z = np.array([2.5, 0.0, 2, 8])
    assert r2_score(y, z) == 0.9486081370449679


def test_r2_score_float_2D() -> None:
    z = np.array([[0, 2], [-1, 2], [8, -5]])
    y = np.array([[0.5, 1], [-1, 1], [7, -6]])
    assert r2_score(y, z) == 0.9512661251791686
