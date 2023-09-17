from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from photoholmes.models.base import BaseMethod
from photoholmes.models.splicebuster.utils import (UNIQUE_TUPLES,
                                                   encode_matrix,
                                                   mahalanobis_distance,
                                                   qround,
                                                   third_order_residual)
from photoholmes.utils.clustering.gaussian_mixture import GaussianMixture
from photoholmes.utils.generic import load_yaml


class Splicebuster(BaseMethod):
    def __init__(
        self, block_size: int = 128, stride: int = 8, q: int = 2, T: int = 1, **kwargs
    ):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.stride = stride
        self.q = q
        self.T = T

    def _quantize(self, x: np.ndarray) -> np.ndarray:
        """
        Uniform quantization used in the paper.
        """
        if isinstance(x, np.ndarray):
            return np.clip(qround(x / self.q) + self.T, 0, 2 * self.T)

    def _fast_cooccurrance_histograms(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate coocurrance histograms efficiently.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"argument 'x' must be of type np.ndarray, not {type(x)}")

        code = (2 * self.T + 1) ** np.arange(4)

        coh = np.zeros(25)
        cov = np.zeros(25)
        coded_matrix_h = encode_matrix(x, code)
        coded_matrix_v = encode_matrix(x, code, axis=1)

        for i, tup in enumerate(UNIQUE_TUPLES):
            tup = np.array(tup)
            coded_tup = np.dot(tup, code)
            coded_tup_simm = np.dot(tup[::-1], code)
            coded_comp_tup = np.dot(2 * self.T - tup, code)
            coded_comp_tup_simm = np.dot((2 * self.T - tup)[::-1], code)
            coh[i] = (
                (
                    (coded_matrix_h == coded_tup)
                    + (coded_matrix_h == coded_tup_simm)
                    + (coded_matrix_h == coded_comp_tup)
                    + (coded_matrix_h == coded_comp_tup_simm)
                )
            ).sum()
            cov[i] = (
                (
                    (coded_matrix_v == coded_tup)
                    + (coded_matrix_v == coded_tup_simm)
                    + (coded_matrix_v == coded_comp_tup)
                    + (coded_matrix_v == coded_comp_tup_simm)
                )
            ).sum()

        return coh / np.sum(coh), cov / np.sum(cov)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run splicebuster on an image."""
        features: List[np.ndarray] = list()
        coords = list()
        pbar = tqdm(
            total=(image.shape[0] // self.stride) * (image.shape[1] // self.stride)
        )
        for i in range(0, image.shape[0], self.stride):
            for j in range(0, image.shape[1], self.stride):
                x = image[i : i + self.block_size, j : j + self.block_size]
                qhres = self._quantize(third_order_residual(x))
                qvres = self._quantize(third_order_residual(x, axis=1))

                Hhh, Hhv = self._fast_cooccurrance_histograms(qhres)
                Hvh, Hvv = self._fast_cooccurrance_histograms(qvres)

                features.append(np.concatenate((Hhh + Hvv, Hhv + Hvh)))
                coords.append((i, j))
                pbar.update(1)
        pbar.close()

        gm = GaussianMixture(2)
        mus, covs = gm.fit(features)

        labels = mahalanobis_distance(features, mus[1], covs[1]) / mahalanobis_distance(
            features, mus[0], covs[0]
        )
        labels_comp = 1 / labels

        heatmap = np.zeros((2, *image.shape))
        for k, (i, j) in enumerate(coords):
            heatmap[0][i : i + self.stride, j : j + self.stride] = labels[k]
            heatmap[1][i : i + self.stride, j : j + self.stride] = labels_comp[k]

        return heatmap

    @classmethod
    def from_config(cls, config: Optional[str | Dict[str, Any]]):
        if isinstance(config, str):
            config = load_yaml(config)
        if config is None:
            config = {}

        return cls(**config)
