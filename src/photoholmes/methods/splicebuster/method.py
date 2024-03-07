# derived from https://www.grip.unina.it/download/prog/Splicebuster/
import warnings
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.linalg import LinAlgWarning
from torch import Tensor

from photoholmes.methods.base import BaseMethod
from photoholmes.postprocessing.resizing import upscale_mask
from photoholmes.utils.generic import load_yaml
from photoholmes.utils.pca import PCA

from .config import WeightConfig
from .utils import (
    encode_matrix,
    feat_reduce_matrix,
    gaussian_mixture_mahalanobis,
    gaussian_uniform_mahalanobis,
    get_saturated_region_mask,
    quantize,
    third_order_residual,
)

warnings.filterwarnings("error", category=LinAlgWarning)


class Splicebuster(BaseMethod):
    """Implementation of the Splicebuster method [Cozzolino et al., 2015].

    This method is based on detecting splicing from features extracted from the image's residuals.

    The original implementation(language: Python) is available at:
    https://www.grip.unina.it/download/prog/Splicebuster/"""

    def __init__(
        self,
        block_size: int = 128,
        stride: int = 8,
        q: int = 2,
        T: int = 1,
        saturation_prob: float = 0.85,
        pca_dim: int = 25,
        mixture: Literal["uniform", "gaussian"] = "uniform",
        pca: Literal["original", "uncentered", "correct"] = "original",
        seed: Union[int, None] = 0,
        weights: Union[WeightConfig, Literal["original"], None] = "original",
        **kwargs,
    ):
        """
        Initializes Splicebuster method class.
        Params:
        - block_size: size of the blocks used for feature extraction.
        - stride: stride used for feature extraction.
        - q: quantization level.
        - T: Truncation level.
        - pca_dim: number of dimensions to keep after PCA. If 0, PCA is not used.
        - seed: random seed for mixture model initialization.
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
        self.saturation_prob = saturation_prob
        self.pca_dim = pca_dim
        self.pca = pca

        self.weight_params: Optional[WeightConfig]
        if weights == "original":
            self.weight_params = WeightConfig()
        else:
            self.weight_params = weights

        self.mahalanobis_estimation = self._init_mahal_estimation(mixture)
        self.seed = seed

    def _init_mahal_estimation(self, mixture: str):
        """
        Obtains the corresponding mahalanobis distance from a mixture model, according to the input 'mixture'.
        Input:
            - mixture: string indicating the mixture model to use. Options: 'uniform', 'gaussian'.
        Output:
            - mahalanobis from features function.
        """
        if mixture == "gaussian":
            return gaussian_mixture_mahalanobis
        elif mixture == "uniform":
            return gaussian_uniform_mahalanobis
        else:
            raise ValueError(
                (
                    f"mixture {mixture} is not a valid mixture model. "
                    'Please select either "uniform" or "gaussian"'
                )
            )

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
    ) -> Tuple[NDArray, NDArray, int, Tuple[NDArray, NDArray]]:
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

        weights = np.zeros((len(x_range), len(y_range)))

        for x_i, i in enumerate(x_range):
            for x_j, j in enumerate(y_range):
                block_weights = mask[i : i + self.stride, j : j + self.stride]
                weights[x_i, x_j] = np.sum(block_weights)

                if weights[x_i, x_j] == 0:
                    continue

                Hhh = np.histogram(
                    qhh[i : i + self.stride, j : j + self.stride],
                    bins=bins,
                    weights=block_weights,
                    density=True,
                )[0].astype(float)
                Hvv = np.histogram(
                    qvv[i : i + self.stride, j : j + self.stride],
                    bins=bins,
                    weights=block_weights,
                    density=True,
                )[0].astype(float)
                Hhv = np.histogram(
                    qhv[i : i + self.stride, j : j + self.stride],
                    bins=bins,
                    weights=block_weights,
                    density=True,
                )[0].astype(float)
                Hvh = np.histogram(
                    qvh[i : i + self.stride, j : j + self.stride],
                    bins=bins,
                    weights=block_weights,
                    density=True,
                )[0].astype(float)

                features[x_i, x_j] = np.concatenate((Hhv + Hvh, Hhh + Hvv))

        weights /= self.stride**2

        return features, weights, feat_dim, (np.array(x_range), np.array(y_range))

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
    ) -> Tuple[NDArray, NDArray, Tuple[NDArray, NDArray]]:
        """
        Computes features, weights and coordinates for an image.
        """
        qhh, qhv, qvh, qvv = self.filter_and_encode(image)

        if self.weight_params is not None:
            mask = get_saturated_region_mask(
                image,
                float(self.weight_params.low_th) / 255,
                float(self.weight_params.high_th) / 255,
            )

            mask = mask[4:-4, 4:-4]
            features, weights, feat_dim, coords = self.compute_histograms(
                qhh, qhv, qvh, qvv, mask
            )
        else:
            features, weights, feat_dim, coords = self.compute_histograms(
                qhh, qhv, qvh, qvv
            )

        strides_x_block = self.block_size // self.stride
        block_features = np.zeros(
            (
                features.shape[0] - strides_x_block + 1,
                features.shape[1] - strides_x_block + 1,
                feat_dim,
            )
        )
        block_weights = np.zeros(
            (
                features.shape[0] - strides_x_block + 1,
                features.shape[1] - strides_x_block + 1,
            )
        )
        for i in range(block_features.shape[0]):
            for j in range(block_features.shape[1]):
                block_weights[i, j] = weights[
                    i : i + strides_x_block, j : j + strides_x_block
                ].mean(axis=(0, 1))
                block_features[i, j] = features[
                    i : i + strides_x_block, j : j + strides_x_block
                ].mean(axis=(0, 1))

                block_features[i, j] /= np.maximum(block_weights[i, j], 1e-20)

        if self.pca_dim > 0:
            block_features = np.sqrt(block_features)

        coords = self.correct_coords(coords)

        return block_features, block_weights, coords

    def _reduce_dimensions(
        self, flat_features: NDArray, valid_features: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """
        Reduces the dimensions of a set of features using PCA, computing it differently
        according to attribute 'pca'.
        """
        if self.pca == "original":
            t = feat_reduce_matrix(self.pca_dim, valid_features)
            flat_features = np.matmul(flat_features, t)
            valid_features = np.matmul(valid_features, t)
        elif self.pca == "uncentered":
            pca = PCA(n_components=self.pca_dim, whiten=True)
            pca.fit(valid_features)
            # apply PCA over uncentered features as original implementation
            flat_features = pca.transform(flat_features + valid_features.mean(axis=0))
            valid_features = pca.transform(flat_features + valid_features.mean(axis=0))
        else:
            pca = PCA(self.pca_dim)
            valid_features = pca.fit_transform(valid_features)
            flat_features = pca.transform(flat_features)
        return flat_features, valid_features

    def predict(self, image: NDArray) -> Dict[str, Tensor]:
        """
        Run splicebuster on an image.
        Params:
            image: normalized image
        Returns:
            heatmap: splicebuster output
        """
        if image.ndim == 3:
            image = image[:, :, 0]
        X, Y = image.shape[:2]

        features, weights, coords = self.compute_features(image)
        valid = weights >= self.saturation_prob
        flat_features = features.reshape(-1, features.shape[-1])
        valid_features = flat_features[valid.flatten()]

        if self.pca_dim > 0:
            flat_features, valid_features = self._reduce_dimensions(
                flat_features, valid_features
            )

        try:
            labels = self.mahalanobis_estimation(
                self.seed, valid_features, flat_features, valid
            )
        except LinAlgWarning:
            labels = np.zeros(flat_features.shape[0])
        heatmap = labels.reshape(features.shape[:2])
        heatmap = heatmap / max(np.max(labels), 1)
        heatmap = upscale_mask(coords, heatmap, (X, Y), method="linear", fill_value=0)

        return {"heatmap": torch.tensor(heatmap, dtype=torch.float32)}

    @classmethod
    def from_config(
        cls, config: Optional[str | Dict[str, Any]], device: Optional[str] = "cpu"
    ):
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

        config["device"] = device

        return cls(**config)
