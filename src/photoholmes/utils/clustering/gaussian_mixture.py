from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from sklearn.mixture import GaussianMixture as sklearn_gmm


class GaussianMixture:
    """Wrapper to use Gaussian Mixtures from scikit-learn library"""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.gm = sklearn_gmm(n_components=n_components)

    def fit(self, features: np.ndarray) -> Tuple[ArrayLike, ArrayLike]:
        """Predicts masks from a list of images."""

        self.gm.fit(features)
        mus = self.gm.means_
        covs = self.gm.covariances_
        return mus, covs
