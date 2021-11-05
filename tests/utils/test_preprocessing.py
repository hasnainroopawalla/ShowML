from showml.utils.preprocessing import normalize
import numpy as np
from numpy.testing import assert_almost_equal


def test_normalize_1D() -> None:
    data = np.array([1, 5, 2, 8, 3])
    expected = np.array([-1.12815215, 0.48349378, -0.72524067, 1.69222822, -0.32232919])
    assert_almost_equal(normalize(data), expected)


def test_normalize_2D() -> None:
    data = np.array([[1, 5, 2, 8, 3], [7, 8, 9, 4, 1], [67, 34, 89, 44, 2]])
    expected = np.array(
        [
            [-0.80538727, -0.8191675, -0.79393478, -0.59299945, 1.22474487],
            [-0.60404045, -0.58877664, -0.61656637, -0.81537425, -1.22474487],
            [1.40942772, 1.40794413, 1.41050115, 1.4083737, 0.0],
        ]
    )
    assert_almost_equal(normalize(data), expected)
