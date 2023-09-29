from typing import List, Tuple

import numpy as np
from numpy.typing import ArrayLike


def baseN_to_base10(t: Tuple, N: int) -> int:
    tN = sum([N**i * x for i, x in enumerate(t)])
    return tN


def get_tuples(T: int) -> List[np.ndarray]:
    tuples: List[np.ndarray] = list()

    n_values = 2 * T + 1
    max_value = 2 * T

    coded_tuples = list()
    for x0 in range(n_values):
        for x1 in range(n_values):
            for x2 in range(n_values):
                for x3 in range(n_values):
                    x = baseN_to_base10((x0, x1, x2, x3), n_values)
                    x_c = baseN_to_base10(
                        (
                            max_value - x0,
                            max_value - x1,
                            max_value - x2,
                            max_value - x3,
                        ),
                        n_values,
                    )
                    x_r = baseN_to_base10((x3, x2, x1, x0), n_values)
                    x_cr = baseN_to_base10(
                        (
                            max_value - x3,
                            max_value - x2,
                            max_value - x1,
                            max_value - x0,
                        ),
                        n_values,
                    )

                    if (
                        x in coded_tuples
                        or x_c in coded_tuples
                        or x_r in coded_tuples
                        or x_cr in coded_tuples
                    ):
                        continue
                    else:
                        tuples.append(np.array((x0, x1, x2, x3)))
                        coded_tuples.append(x)
    return tuples


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
