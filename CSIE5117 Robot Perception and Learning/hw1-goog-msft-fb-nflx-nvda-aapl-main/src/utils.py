import numpy as np


def to_scalar(x) -> float:
    """Convert a 0-d / 1x1 ndarray-like to a Python float."""
    return np.asarray(x, dtype=float).reshape(()).item()


def to_vec(x) -> np.ndarray:
    """Convert input to a 1-D vector with shape (n,)."""
    return np.asarray(x, dtype=float).reshape(-1)


def row1(x) -> np.ndarray:
    """Convert input to a single-row array with shape (1, n)."""
    return np.asarray(x, dtype=float).reshape(1, -1)
