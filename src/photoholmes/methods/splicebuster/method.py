# derived from https://www.grip.unina.it/download/prog/Splicebuster/
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from photoholmes.methods.base import BaseMethod
from photoholmes.utils.clustering.gaussian_mixture import GaussianMixture
from photoholmes.utils.clustering.gaussian_uniform import GaussianUniformEM
from photoholmes.utils.generic import load_yaml
from photoholmes.utils.pca import PCA
from photoholmes.utils.postprocessing.resizing import upscale_mask

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

        self.weight_params: Optional[WeightConfig]
        if weights == "original":
            self.weight_params = WeightConfig()
        else:
            self.weight_params = weights

        self.mixture = mixture

    def filter_and_encode(
        self, image: NDArray
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Apply third order residual filtering, quantization, and
        encode the result as base 3 integers for fast coocurrance counting.
        """
        qh_res = quantize(third_order_residual(image), self.T, self.q)
        qv_res = quantize(third_order_residual(image, axis=1), self.T, self.q)

        qhh = encode_matrix(qh_res, T=self.T)
        qhv = encode_matrix(qh_res, T=self.T, axis=1)
        qvh = encode_matrix(qv_res, T=self.T)
        qvv = encode_matrix(qv_res, T=self.T, axis=1)

        return qhh, qhv, qvh, qvv

    def compute_histograms(
        self,
        qhh: NDArray[np.int64],
        qhv: NDArray[np.int64],
        qvh: NDArray[np.int64],
        qvv: NDArray[np.int64],
        mask: Optional[NDArray] = None,
    ) -> Tuple[NDArray, int, Tuple[NDArray, NDArray]]:
        """
        Efficiently compute histograms for stride x stride blocks.
        """
        H, W = qhh.shape
        x_range = np.arange(0, H - self.stride + 1, self.stride)
        y_range = np.arange(0, W - self.stride + 1, self.stride)

        if mask is None:
            mask = np.ones((H, W), dtype=np.uint8)

        n_bins = 1 + np.max((qhh, qhv, qvh, qvv))
        bins = np.arange(0, n_bins + 1)
        feat_dim = int(2 * n_bins)
        features = np.zeros((len(x_range), len(y_range), feat_dim))

        for x_i, i in enumerate(x_range):
            for x_j, j in enumerate(y_range):
                block_weights = mask[i : i + self.stride, j : j + self.stride]

                Hhh = np.histogram(
                    qhh[i : i + self.stride, j : j + self.stride],
                    bins=bins,
                    weights=block_weights,
                )[0].astype(float)
                Hvv = np.histogram(
                    qhv[i : i + self.stride, j : j + self.stride],
                    bins=bins,
                    weights=block_weights,
                )[0].astype(float)
                Hhv = np.histogram(
                    qvh[i : i + self.stride, j : j + self.stride],
                    bins=bins,
                    weights=block_weights,
                )[0].astype(float)
                Hvh = np.histogram(
                    qvh[i : i + self.stride, j : j + self.stride],
                    bins=bins,
                    weights=block_weights,
                )[0].astype(float)

                features[x_i, x_j] = np.concatenate((Hhh + Hvv, Hhv + Hvh)) / 2

        return features, feat_dim, (np.array(x_range), np.array(y_range))

    def correct_coords(
        self, coords: Tuple[NDArray, NDArray]
    ) -> Tuple[NDArray, NDArray]:
        """
        Apply correction to coordinates to account for the window filtering,
        coocurrance computation and center coordinate on window.
        """
        x_coords, y_coords = coords
        # window filtering
        x_coords += 4
        y_coords += 4
        # center coordinate on window
        x_coords = x_coords + (self.stride - 1) / 2
        y_coords = y_coords + (self.stride - 1) / 2

        # moving average compensation
        stride_x_block = self.block_size // self.stride
        low = int(np.floor((stride_x_block - 1) / 2))
        high = int(np.ceil((stride_x_block - 1) / 2))
        x_coords = (x_coords[low:-high] + x_coords[high:-low]) / 2
        y_coords = (y_coords[low:-high] + y_coords[high:-low]) / 2

        return x_coords, y_coords

    def compute_features(
        self, image: NDArray
    ) -> Tuple[NDArray, Tuple[NDArray, NDArray]]:
        qhh, qhv, qvh, qvv = self.filter_and_encode(image)

        if self.weight_params is not None:
            mask = get_saturated_region_mask(
                image,
                self.weight_params.low_th / 255,
                self.weight_params.high_th / 255,
                self.weight_params.opening_kernel_radius,
                self.weight_params.dilation_kernel_size,
            )
            mask = mask[2:-5, 2:-5]
            features, feat_dim, coords = self.compute_histograms(
                qhh, qhv, qvh, qvv, mask
            )
        else:
            features, feat_dim, coords = self.compute_histograms(qhh, qhv, qvh, qvv)

        strides_x_block = self.block_size // self.stride
        block_features = np.zeros(
            (
                features.shape[0] - strides_x_block + 1,
                features.shape[1] - strides_x_block + 1,
                feat_dim,
            )
        )
        for i in range(block_features.shape[0]):
            for j in range(block_features.shape[1]):
                block_features[i, j] = features[
                    i : i + strides_x_block, j : j + strides_x_block
                ].sum(axis=(0, 1))
                block_features[i, j] /= max(np.sum(block_features[i, j]), 1e-20)

        coords = self.correct_coords(coords)

        if self.pca_dim > 0:
            block_features = np.sqrt(block_features)

        return block_features, coords

    def predict(self, image: NDArray) -> NDArray:
        """Run splicebuster on an image.
        Params:
            image: normalized image
        Returns:
            heatmap: splicebuster output
        """
        X, Y = image.shape[:2]

        features, coords = self.compute_features(image)
        flat_features = features.reshape(-1, features.shape[-1])

        if self.pca_dim > 0:
            pca = PCA(n_components=self.pca_dim)
            flat_features = pca.fit_transform(flat_features)

        if self.mixture == "gaussian":
            gg_mixt = GaussianMixture()
            mus, covs = gg_mixt.fit(flat_features)
            labels = mahalanobis_distance(
                flat_features, mus[1], covs[1]
            ) / mahalanobis_distance(flat_features, mus[0], covs[0])
            labels_comp = 1 / labels
            labels = labels if labels.sum() < labels_comp.sum() else labels_comp
        elif self.mixture == "uniform":
            gu_mixt = GaussianUniformEM()
            mus, covs, _ = gu_mixt.fit(flat_features)
            _, labels = gu_mixt.predict(flat_features)
        else:
            raise ValueError(
                (
                    f"mixture {self.mixture} is not a valid mixture model. "
                    'Please select either "uniform" or "gaussian"'
                )
            )

        heatmap = labels.reshape(features.shape[:2])

        heatmap = heatmap / np.max(labels)
        heatmap = upscale_mask(coords, heatmap, (X, Y), method="linear", fill_value=0)
        return heatmap

    @classmethod
    def from_config(cls, config: Optional[str | Dict[str, Any]]):
        """
        Instantiate the model from configuration dictionary or yaml.

        Params:
            config: path to the yaml configuration or a dictionary with
                    the parameters for the model.
        """
        if isinstance(config, str):
            config = load_yaml(config)

        if config is None:
            config = {}

        if "weights" in config:
            config["weights"] = WeightConfig(**config["weights"])

        return cls(**config)
