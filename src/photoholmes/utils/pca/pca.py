from typing import List, Union

import numpy as np
from sklearn.decomposition import PCA as sklearn_pca


class PCA:
    """Wrapper to use PCA from scikit-learn library"""

    def __init__(self, n_components: int = 25, whiten: bool = True):
        self.n_components = n_components
        self.pca = sklearn_pca(n_components=n_components, whiten=whiten)

    def fit(self, features: Union[np.ndarray, List[np.ndarray]]):
        self.pca.fit(features)  # type: ignore

    def transform(self, features: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        return self.pca.transform(features)  # type: ignore

    def fit_transform(
        self, features: Union[np.ndarray, List[np.ndarray]]
    ) -> np.ndarray:
        return self.pca.fit_transform(features)  # type: ignore

    def get_covariance(self):
        return self.pca.get_covariance()
