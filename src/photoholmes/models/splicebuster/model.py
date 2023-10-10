# derived from https://www.grip.unina.it/download/prog/Splicebuster/
from typing import Literal, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from photoholmes.models.base import BaseMethod
from photoholmes.utils.clustering.gaussian_mixture import GaussianMixture
from photoholmes.utils.clustering.gaussian_uniform import GaussianUniformEM
from photoholmes.utils.pca import PCA

from .config import WeightConfig
from .utils import (
    encode_matrix,
    get_saturated_region_mask,
    mahalanobis_distance,
    quantize,
    third_order_residual,
)


class Splicebuster(BaseMethod):
    def __init__(
        self,
        block_size: int = 128,
        stride: int = 8,
        q: int = 2,
        T: int = 1,
        pca_dim: int = 25,
        mixture: Literal["uniform", "gaussian"] = "uniform",
        weights: Union[WeightConfig, Literal["original"], None] = None,
        **kwargs,
    ):
        """
        Splicebuster implementation.
        Params:
        - block_size: size of the blocks used for feature extraction.
        - stride: stride used for feature extraction.
        - q: quantization level.
        - T: Truncation level.
        - pca_dim: number of dimensions to keep after PCA. If 0, PCA is not used.
        - weight_params: provides parameters for weighted feature computation.Options:
            - None: do not use weights.
            - "original": use parameters from the original implementation.
            - WeightConfig object: use custom parameters.
        """
        super().__init__(**kwargs)
        self.block_size = block_size
        self.stride = stride
        self.q = q
        self.T = T
        self.pca_dim = pca_dim
        self.mixture = mixture
        if weights == "original":
            self.weight_params = WeightConfig()
        else:
            self.weight_params = weights

    def filter_and_encode(
        self, image: NDArray
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Apply third order residual filtering, quantization, and
        encode the result as base 3 integers for fast coocurrance counting.
        """
        qh_res = quantize(third_order_residual(image), self.T, self.q)
        qv_res = quantize(third_order_residual(image, axis=1), self.T, self.q)

        qhh = encode_matrix(qh_res)
        qhv = encode_matrix(qh_res, axis=1)
        qvh = encode_matrix(qv_res)
        qvv = encode_matrix(qv_res, axis=1)

        return qhh, qhv, qvh, qvv

    def compute_weighted_histograms(
        self,
        mask: NDArray,
        qhh: NDArray[np.int16],
        qhv: NDArray[np.int16],
        qvh: NDArray[np.int16],
        qvv: NDArray[np.int16],
    ) -> Tuple[NDArray, int]:
        """
        Efficiently compute weighted histogram for stride x stride blocks.
        """
        H, W = qhh.shape
        x_range = range(0, H - self.stride + 1, self.stride)
        y_range = range(0, W - self.stride + 1, self.stride)

        n_bins = 1 + np.max((qhh, qhv, qvh, qvv))
        feat_dim = int(2 * n_bins)
        features = np.zeros((len(x_range), len(y_range), feat_dim))

        for x_i, i in enumerate(x_range):
            for x_j, j in enumerate(y_range):
                block_weights = mask[i : i + self.stride, j : j + self.stride]

                Hhh = np.histogram(
                    qhh[i : i + self.stride, j : j + self.stride],
                    bins=n_bins,
                    weights=block_weights,
                )[0].astype(float)
                Hvv = np.histogram(
                    qhv[i : i + self.stride, j : j + self.stride],
                    bins=n_bins,
                    weights=block_weights,
                )[0].astype(float)
                Hhv = np.histogram(
                    qvh[i : i + self.stride, j : j + self.stride],
                    bins=n_bins,
                    weights=block_weights,
                )[0].astype(float)
                Hvh = np.histogram(
                    qvh[i : i + self.stride, j : j + self.stride],
                    bins=n_bins,
                    weights=block_weights,
                )[0].astype(float)

                features[x_i, x_j] = np.concatenate((Hhh + Hvv, Hhv + Hvh)) / 2

        return features, feat_dim

    def compute_histograms(
        self,
        qhh: NDArray[np.int16],
        qhv: NDArray[np.int16],
        qvh: NDArray[np.int16],
        qvv: NDArray[np.int16],
    ) -> Tuple[NDArray, int]:
        """
        Efficiently compute histograms for stride x stride blocks.
        """
        H, W = qhh.shape
        x_range = range(0, H - self.stride + 1, self.stride)
        y_range = range(0, W - self.stride + 1, self.stride)

        n_bins = 1 + np.max((qhh, qhv, qvh, qvv))
        feat_dim = int(2 * n_bins)
        features = np.zeros((len(x_range), len(y_range), feat_dim))

        for x_i, i in enumerate(x_range):
            for x_j, j in enumerate(y_range):
                Hhh = np.histogram(
                    qhh[i : i + self.stride, j : j + self.stride], bins=n_bins
                )[0].astype(float)
                Hvv = np.histogram(
                    qhv[i : i + self.stride, j : j + self.stride], bins=n_bins
                )[0].astype(float)
                Hhv = np.histogram(
                    qvh[i : i + self.stride, j : j + self.stride], bins=n_bins
                )[0].astype(float)
                Hvh = np.histogram(
                    qvv[i : i + self.stride, j : j + self.stride], bins=n_bins
                )[0].astype(float)

                features[x_i, x_j] = np.concatenate((Hhh + Hvv, Hhv + Hvh)) / 2
        return features, feat_dim

    def compute_features(self, image: NDArray) -> NDArray:
        qhh, qhv, qvh, qvv = self.filter_and_encode(image)

        if self.weight_params is not None:
            mask = get_saturated_region_mask(
                image,
                self.weight_params.low_th,
                self.weight_params.high_th,
                self.weight_params.opening_kernel_radius,
                self.weight_params.dilation_kernel_size,
            )
            mask = mask[2:-5, 2:-5]
            features, feat_dim = self.compute_weighted_histograms(
                mask, qhh, qhv, qvh, qvv
            )
        else:
            features, feat_dim = self.compute_histograms(qhh, qhv, qvh, qvv)

        strides_x_block = self.block_size // self.stride
        block_features = np.zeros(
            (
                features.shape[0] - strides_x_block,
                features.shape[1] - strides_x_block,
                feat_dim,
            )
        )
        for i in range(block_features.shape[0]):
            for j in range(block_features.shape[1]):
                block_features[i, j] = features[
                    i : i + strides_x_block, j : j + strides_x_block
                ].sum(axis=(0, 1))
                block_features[i, j] /= max(np.sum(block_features[i, j]), 1e-20)

        if self.pca_dim > 0:
            block_features = np.sqrt(block_features)

        return block_features

    def predict(self, image: NDArray) -> NDArray:
        """Run splicebuster on an image.
        Params:
            image: normalized image
        Returns:
            heatmap: splicebuster output
        """

        features = self.compute_features(image)
        flat_features = features.reshape(-1, features.shape[-1])

        if self.pca_dim > 0:
            pca = PCA(n_components=self.pca_dim)
            flat_features = pca.fit_transform(flat_features)

        if self.mixture == "gaussian":
            mixt = GaussianMixture()
            mus, covs = mixt.fit(flat_features)
            labels = mahalanobis_distance(
                flat_features, mus[1], covs[1]
            ) / mahalanobis_distance(flat_features, mus[0], covs[0])
            labels_comp = 1 / labels
            labels = labels if labels.sum() < labels_comp.sum() else labels_comp
        elif self.mixture == "uniform":
            mixt = GaussianUniformEM()
            mus, covs, _ = mixt.fit(flat_features)
            _, labels = mixt.predict(flat_features)
        else:
            raise ValueError(
                (
                    f"mixture {self.mixture} is not a valid mixture model. "
                    'Please select either "uniform" or "gaussian"'
                )
            )

        heatmap = np.empty(
            (image.shape[0] - self.block_size, image.shape[1] - self.block_size)
        )
        heatmap = labels.reshape(features.shape[:2])

        heatmap = heatmap / np.max(labels)
        return heatmap
