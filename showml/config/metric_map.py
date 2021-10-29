from typing import Callable, Dict
from showml.utils.metrics import (
    accuracy,
    mean_square_error,
    r2_score,
    binary_cross_entropy,
)

metric_map: Dict[str, Callable] = {
    "accuracy": accuracy,
    "mean_square_error": mean_square_error,
    "r2_score": r2_score,
    "binary_cross_entropy": binary_cross_entropy,
}
