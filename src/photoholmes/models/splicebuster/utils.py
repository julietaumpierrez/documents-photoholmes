from typing import Union

import numpy as np
from numpy.typing import ArrayLike


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


def quantize(x: np.ndarray, T: int = 1, q: Union[int, float] = 2) -> np.ndarray:
    """
    Uniform quantization used in the paper.
    """
    q = 3 * float(q) / 256
    if isinstance(x, np.ndarray):
        return np.clip(qround(x / q) + T, 0, 2 * T)


def encode_matrix(X: np.ndarray, T: int = 1, k: int = 4, axis=0) -> np.ndarray:
    coded_shape = (X.shape[0] - k + 1, X.shape[1] - k + 1)

    encoded_matrix = np.zeros((2, *coded_shape))
    base = 2 * T + 1
    max_value = (2 * T + 1) ** k - 1

    if axis == 0:
        for i, b in enumerate(range(k)):
            encoded_matrix[0] += base**b * X[: -k + 1, i : coded_shape[1] + i]
            encoded_matrix[1] += (
                base ** (k - 1 - b) * X[: -k + 1, i : coded_shape[1] + i]
            )

    elif axis == 1:
        for i, b in enumerate(range(k)):
            encoded_matrix[0] += base**b * X[i : coded_shape[0] + i, : -k + 1]
            encoded_matrix[1] += (
                base ** (k - 1 - b) * X[i : coded_shape[0] + i, : -k + 1]
            )

    reduced = np.minimum(encoded_matrix[0], encoded_matrix[1])
    reduced = np.minimum(reduced, max_value - encoded_matrix[0])
    reduced = np.minimum(reduced, max_value - encoded_matrix[1])

    _, reduced = np.unique(reduced, return_inverse=True)
    return reduced.reshape(coded_shape)


def mahalanobis_distance(X: ArrayLike, mu: ArrayLike, cov: ArrayLike):
    X = np.array(X)
    cov = np.array(cov)
    inv_cov = np.linalg.inv(cov)
    mu = np.array(mu)

    X_ = np.empty(X.shape[0], dtype=np.float16)
    X_centered = X - mu
    for i in range(X_.shape[0]):
        X_[i] = np.sqrt(X_centered[i] @ inv_cov @ X_centered[i].T)
    return X_
