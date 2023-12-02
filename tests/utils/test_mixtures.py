import numpy as np
import pytest

from photoholmes.utils.clustering.gaussian_mixture import GaussianMixture
from photoholmes.utils.clustering.gaussian_uniform import GaussianUniformEM


# =========================== Guassian Mixture =========================================
class TestGaussianMixture:
    @pytest.fixture
    def gm(self):
        return GaussianMixture(n_components=2)

    def test_init(self, gm):
        assert gm.n_components == 2
        assert hasattr(gm, "gm")

    def test_fit(self, gm):
        # Create a numpy array
        features = np.random.rand(100, 10)

        # Fit the GaussianMixture
        mus, covs = gm.fit(features)

        # Check that the means and covariances have the right shape
        assert mus.shape == (2, 10)
        assert covs.shape == (2, 10, 10)


# =========================== Guassian Uniform =========================================
class TestGaussianUniformEM:
    @pytest.fixture
    def gu(self):
        return GaussianUniformEM(n_init=2)

    def test_init(self, gu: GaussianUniformEM):
        assert gu.n_init == 2
        assert hasattr(gu, "pi")

    def test_fit(self, gu: GaussianUniformEM):
        # Create a numpy array
        X1 = np.random.rand(100, 10)
        X2 = np.random.uniform(0, 1, (100, 10))
        features = np.vstack((X1, X2))

        # Fit the GaussianUniformEM
        mean, covariance_matrix, pi = gu.fit(features)

        # Check that the mean, covariance_matrix, and pi have the right shape
        assert mean.shape == (10,)
        assert covariance_matrix.shape == (10, 10)
        assert isinstance(pi, float)

    def test_predict(self, gu: GaussianUniformEM):
        # Create a numpy array
        X1 = np.random.rand(100, 10)
        X2 = np.random.uniform(0, 1, (100, 10))
        features = np.vstack((X1, X2))

        # Fit the GaussianUniformEM
        gu.fit(features)

        # Predict the class of the samples
        gammas, mahal = gu.predict(features)

        # Check that the gammas and mahal have the right shape
        assert gammas.shape == (200,)
        assert mahal.shape == (200,)
