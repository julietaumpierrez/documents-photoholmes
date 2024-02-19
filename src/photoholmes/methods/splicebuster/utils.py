from functools import reduce
from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation, binary_opening


def third_order_residual(x: NDArray, axis: int = 0) -> NDArray:
    """
    Calculates the third order residual as specified in the paper.
    Params:
    - x: input image. Dims (H, W)
    - axis: axis along which to calculate the residual. 0 for horizontal,
            1 for vertical

    Returns:
    - residual: third order residual. Dims (H - 4, W - 4)
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


def qround(y: NDArray) -> NDArray:
    return np.sign(y) * np.floor(np.abs(y) + 0.5)


def quantize(x: NDArray, T: int = 1, q: Union[int, float] = 2) -> NDArray[np.int8]:
    """
    Uniform quantization used in the paper.
    """
    q = 3 * float(q) / 256
    if isinstance(x, np.ndarray):
        return np.clip(qround(x / q) + T, 0, 2 * T)


def encode_matrix(
    X: NDArray[np.int8], T: int = 1, c: int = 4, axis: int = 0
) -> NDArray[np.int16]:
    coded_shape = (X.shape[0] - c, X.shape[1] - c)

    encoded_matrix = np.zeros((2, *coded_shape))
    base = 2 * T + 1
    max_value = (2 * T + 1) ** c - 1

    left_index = c // 2
    if axis == 0:
        for i, b in enumerate(range(c)):
            encoded_matrix[0] += (
                base**b * X[left_index : -c + left_index, i : coded_shape[1] + i]
            )
            encoded_matrix[1] += (
                base ** (c - 1 - b)
                * X[left_index : -c + left_index, i : coded_shape[1] + i]
            )

    elif axis == 1:
        for i, b in enumerate(range(c)):
            encoded_matrix[0] += (
                base**b * X[i : coded_shape[0] + i, left_index : -c + left_index]
            )
            encoded_matrix[1] += (
                base ** (c - 1 - b)
                * X[i : coded_shape[0] + i, left_index : -c + left_index]
            )

    reduced = np.minimum(encoded_matrix[0], encoded_matrix[1])
    reduced = np.minimum(reduced, max_value - encoded_matrix[0])
    reduced = np.minimum(reduced, max_value - encoded_matrix[1])

    _, reduced = np.unique(reduced, return_inverse=True)
    return reduced.reshape(coded_shape)


def mahalanobis_distance(X: NDArray, mu: NDArray, cov: NDArray):
    inv_cov = np.linalg.inv(cov)

    X_ = np.empty(X.shape[0], dtype=np.float16)
    X_centered = X - mu
    for i in range(X_.shape[0]):
        X_[i] = np.sqrt(X_centered[i] @ inv_cov @ X_centered[i].T)
    return X_


def get_disk_kernel(radius: int) -> np.ndarray:
    """
    Returns a disk kernel of the given radius.

    Params:
        radius: radius of the disk.
    Returns:
        kernel: disk kernel of the given radius.
    """
    xx, yy = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    kernel = (xx**2 + yy**2) <= radius**2
    return 1.0 * kernel


def get_saturated_region_mask(
    img: np.ndarray,
    low_th: float = 6 / 255,
    high_th: float = 252 / 255,
    opening_kernel_radius: int = 3,
    dilation_kernel_size: int = 9,
) -> NDArray:
    """
    Creates a binary mask of the saturated regions in the image.

    Params:
        - img: image to get the mask from.
        - low_th: lower threshold for the saturated pixels.
        - high_th: upper threshold for the saturated pixels.
        - opening_kernel_radius: radius of the disk kernel used for the opening
            operation.
        - dilation_kernel_size: size of the square kernel used for the dilation
            operation on the final mask.
    Returns:
        - mask: binary mask of the saturated regions in the image. If applied to the
              image, the saturated regions will be set to 0.
    """
    if img.ndim == 2:
        img = img[:, :, None]
    img = img.transpose(2, 0, 1)
    kernel = get_disk_kernel(opening_kernel_radius)
    mask_low = np.array(
        [binary_opening(ch < low_th, kernel) for ch in img], dtype=np.float32
    )
    mask_high = np.array(
        [binary_opening(ch > high_th, kernel) for ch in img], dtype=np.float32
    )
    mask = mask_low + mask_high
    mask = reduce(np.logical_or, mask)
    mask = binary_dilation(mask, np.ones((dilation_kernel_size, dilation_kernel_size)))
    return ~mask * 1.0
