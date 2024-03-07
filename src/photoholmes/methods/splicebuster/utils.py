from functools import reduce
from typing import Union

import numpy as np
import skimage.morphology as ski
from numpy.typing import NDArray

from photoholmes.utils.clustering.gaussian_mixture import GaussianMixture
from photoholmes.utils.clustering.gaussian_uniform import GaussianUniformEM


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
    low_th: float = 6 / 256,
    high_th: float = 252 / 256,
    erotion_kernel_size: int = 9,
):
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
    kernel_high = get_disk_kernel(2)
    kernel_low = get_disk_kernel(3)
    mask_low = np.array(
        [ski.binary_opening(ch < low_th, kernel_low) for ch in img],
        dtype=np.float32,
    )
    mask_high = np.array(
        [ski.binary_opening(ch > high_th, kernel_high) for ch in img],
        dtype=np.float32,
    )
    mask = mask_low + mask_high
    mask = np.logical_not(reduce(np.logical_or, mask))
    mask = ski.binary_erosion(
        mask, np.ones((erotion_kernel_size, erotion_kernel_size), dtype=bool)
    )
    return mask * 1.0


def feat_reduce_matrix(pca_dim: int, X: NDArray, whitten: bool = True) -> NDArray:
    """
    Calculates the matrix transformation to reduce the features using PCA.

    Params:
        - pca_dim (int): number of principal components to keep.
        - X (NDArray): input features. Dims (n_samples, n_features)
        - whitten (bool): whether to whitten the features or not.
    Returns:
        - NDArray: eigenvector matrix to project the features into the reduced space.
    """
    inds = np.arange(pca_dim)
    cov = np.cov(X, rowvar=False, bias=True)
    w, v = np.linalg.eigh(cov)
    w = w[::-1]
    v = v[:, ::-1]
    v = v[:, inds]

    if whitten:
        v = v / np.sqrt(w[inds])

    return v


def gaussian_mixture_mahalanobis(
    seed: Union[None, int],
    valid_features: NDArray,
    flat_features: NDArray,
    valid: NDArray,
) -> NDArray:
    """
    Gaussian Mixture model fit over valid features and prediction of the mahalanobis distance.
    Non-valid features are labeled as 0.
    Inputs:
        - seed: seed for the random number generator.
        - valid_features: non-saturated (flat) features.
        - flat_features: all (flat) features.
        - valid: mask of the valid pixels.
    Output:
        - labels: mahalanobis distance labels.
    """
    gg_mixt = GaussianMixture(seed=seed)
    mus, covs = gg_mixt.fit(valid_features)
    labels = mahalanobis_distance(
        flat_features, mus[1], covs[1]
    ) / mahalanobis_distance(flat_features, mus[0], covs[0])
    labels_comp = 1 / labels
    labels[~valid.flatten()] = 0
    labels_comp[~valid.flatten()] = 0
    labels = labels if labels.sum() < labels_comp.sum() else labels_comp
    labels[~valid.flatten()] = 0
    return labels


def gaussian_uniform_mahalanobis(
    seed: Union[None, int],
    valid_features: NDArray,
    flat_features: NDArray,
    valid: NDArray,
) -> NDArray:
    """
    Gaussian-Uniform model fit over valid features and prediction of the mahalanobis distance.
    Non-valid features are labeled as 0.
    Inputs:
        - seed: seed for the random number generator.
        - valid_features: non-saturated (flat) features.
        - flat_features: all (flat) features.
        - valid: mask of the valid pixels.
    Output:
        - labels: mahalanobis distance labels.
    """
    gu_mixt = GaussianUniformEM(seed=seed)
    mus, covs, _ = gu_mixt.fit(valid_features)
    _, labels = gu_mixt.predict(flat_features)
    labels[~valid.flatten()] = 0
    return labels
