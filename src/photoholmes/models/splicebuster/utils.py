from typing import List

import numpy as np
from numpy.typing import ArrayLike

UNIQUE_TUPLES: List[tuple] = [
    (0, 0, 0, 0),
    (0, 0, 0, 1),
    (0, 0, 0, 2),
    (0, 0, 1, 0),
    (0, 0, 1, 1),
    (0, 0, 1, 2),
    (0, 0, 2, 0),
    (0, 0, 2, 1),
    (0, 0, 2, 2),
    (0, 1, 0, 1),
    (0, 1, 0, 2),
    (0, 1, 1, 0),
    (0, 1, 1, 1),
    (0, 1, 1, 2),
    (0, 1, 2, 0),
    (0, 1, 2, 1),
    (0, 2, 0, 1),
    (0, 2, 0, 2),
    (0, 2, 1, 1),
    (0, 2, 2, 0),
    (0, 2, 2, 1),
    (1, 0, 0, 1),
    (1, 0, 1, 1),
    (1, 0, 2, 1),
    (1, 1, 1, 1),
]


def third_order_residual(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Calculates the third order residual as specified in the paper.
    """
    x = x.astype(np.float32)
    if axis == 0:
        residual = x[:, :-3] - 3 * x[:, 1:-2] + 3 * x[:, 2:-1] - x[:, 3:]
        residual = residual[2:-2, 1:]
    elif axis == 1:
        residual = x[:-3] - 3 * x[1:-2] + 3 * x[2:-1] - x[3:]
        residual = residual[1:, 2:-2]
    else:
        raise ValueError("axis must be 0 (horizontal) or 1 (vertical)")

    return residual


def qround(y: np.ndarray) -> np.ndarray:
    return np.sign(y) * np.floor(np.abs(y) + 0.5)


def encode_matrix(m: np.ndarray, code: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Trick to speed up coocurrances matrix, by taking advantage of the quantized matrix,
    returns a matrix where m[i,j] is the tuple (i:i+4, j) (or (i, j:j+4) depending
    on axis) decoded as a N-base integer.
    """
    code_size = code.shape[0]
    coded_shape = (m.shape[0] - code_size + 1, m.shape[1] - code_size + 1)
    if isinstance(m, np.ndarray):
        coded_matrix = np.zeros(coded_shape)
    else:
        raise TypeError(f"m must be of type np.ndarray, not {type(m)}")

    if axis == 0:
        for i, k in enumerate(code):
            coded_matrix += k * m[: -code_size + 1, i : coded_shape[1] + i]
    elif axis == 1:
        for i, k in enumerate(code):
            coded_matrix += k * m[i : coded_shape[0] + i, : -code_size + 1]
    else:
        raise ValueError("axis must be 0 (horizontal) or 1 (vertical)")

    return coded_matrix


def mahalanobis_distance(X: ArrayLike, m: ArrayLike, C: ArrayLike):
    X = np.array(X)
    C = np.array(C)
    m = np.array(m)

    X_ = np.empty(X.shape[0])
    for i in range(X_.shape[0]):
        x = X[i].reshape(1, -1)
        X_[i] = np.sqrt((x - m) @ np.linalg.inv(C) @ (x - m).T)
    return X_
