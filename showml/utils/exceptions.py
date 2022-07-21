class DatasetSizeMismatchError(Exception):
    """Raised when X and y have different lengths."""


class DataTypeError(Exception):
    """Raised when a variable has an unexpected datatype."""


class InvalidShapeError(Exception):
    """Raised when an unexpected array shape is encountered."""


class InvalidValueError(Exception):
    """Raised when an unexpected value is encountered."""
