import numpy as np

from photoholmes.models.base import BaseMethod
from photoholmes.models.splicebuster.utils import (encode_matrix,
                                                   mahalanobis_distance,
                                                   quantize,
                                                   third_order_residual)
from photoholmes.utils.clustering.gaussian_mixture import GaussianMixture


class Splicebuster(BaseMethod):
    def __init__(
        self, block_size: int = 128, stride: int = 8, q: int = 2, T: int = 1, **kwargs
    ):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.stride = stride
        self.q = q
        self.T = T

    def compute_features(self, image: np.ndarray) -> np.ndarray:
        H, W = image.shape

        qh_res = quantize(third_order_residual(image), self.T, self.q)
        qv_res = quantize(third_order_residual(image, axis=1), self.T, self.q)

        qhh = encode_matrix(qh_res)
        qhv = encode_matrix(qh_res, axis=1)
        qvh = encode_matrix(qv_res)
        qvv = encode_matrix(qv_res, axis=1)

        x_range = range(0, H - self.stride + 1, self.stride)
        y_range = range(0, W - self.stride + 1, self.stride)

        n_bins = 1 + np.max((qhh, qhv, qvh, qvv))
        feat_dim = 2 * n_bins
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
                block_features[i, j] /= np.sum(block_features[i, j])

        return block_features

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run splicebuster on an image."""
        print("Computing features")
        features = self.compute_features(image)
        flat_features = features.reshape(-1, features.shape[-1])
        print(flat_features.shape)

        print("Fitting gaussian mixture")
        gmm = GaussianMixture()
        mus, covs = gmm.fit(flat_features)

        print("Calculating labels")
        labels = mahalanobis_distance(
            flat_features, mus[1], covs[1]
        ) / mahalanobis_distance(flat_features, mus[0], covs[0])
        labels_comp = 1 / labels
        labels = labels if labels.sum() < labels_comp.sum() else labels_comp

        heatmap = np.empty(
            (image.shape[0] - self.block_size, image.shape[1] - self.block_size)
        )
        n_label = 0
        for i in range(0, image.shape[0] - self.block_size, self.stride):
            for j in range(0, image.shape[1] - self.block_size, self.stride):
                heatmap[i : i + self.stride, j : j + self.stride] = labels[n_label]
                n_label += 1

        heatmap = heatmap / np.max(labels)
        return heatmap
