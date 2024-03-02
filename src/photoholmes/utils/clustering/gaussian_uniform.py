# Code derived from https://github.com/grip-unina/noiseprint and CODIGO MARINA
# FIXME: add license a Marina and Co.
from typing import Tuple

import numpy as np
import scipy as sp
from numpy.typing import NDArray

ATTEMPTS = "data/debug/splicebuster/attempts/"
GROUND_TRUTHS = "data/debug/splicebuster/ground-truths/"


def checkpoint(array, array_name: str, load_gt: bool = True):
    np.save(ATTEMPTS + array_name, array)
    return np.load(GROUND_TRUTHS + array_name) if load_gt else array


class GaussianUniformEM:
    """Class to perform Gaussian Uniform Expectation Maximization algorithm."""

    def __init__(
        self,
        p_outlier_init: float = 1e-2,
        outlier_nlogl: int = 42,
        tol: float = 1e-5,
        max_iter: int = 100,
        n_init: int = 30,
        debug_series={},
        seed=None,
    ) -> None:
        """
        Gaussian Uniform Expectation Maximization algorithm.
        Params:
        - p_outlier_init: initial probability of being falsified
        - outlier_nlogl:  log-likelihood of being falsified
        - tol: tolerance used in a single run of the expectation step
        - max_iter: maximum number of iterations in a single run of the expectation step
        - n_init: number of iterations of EM to run
        """
        self.p_outlier_init = p_outlier_init
        self.outlier_nlogl = outlier_nlogl
        self.pi: float = 1 - p_outlier_init
        self.debug_series = debug_series
        self.max_iter = max_iter
        self.tol = tol
        assert n_init > 1, "n_init must be at least 1"
        self.n_init = n_init

        self.random_state = np.random.RandomState(seed)
        self.covariance_matrix: NDArray
        self.mean: NDArray
        self.debug_amount_of_fits = 0
        self.debug_amount_of_random_inits = 0

    def fit(self, X: NDArray) -> Tuple[NDArray, NDArray, float]:
        """
        Fit the model to the data.
        """
        best_loss = self._fit_once(X)
        save = self.mean, self.covariance_matrix, self.pi
        for i in range(self.n_init - 1):
            loss = self._fit_once(X)
            if loss > best_loss:
                best_loss = loss
                save = self.mean, self.covariance_matrix, self.pi
        self.mean, self.covariance_matrix, self.pi = save
        return save

    def _fit_once(self, X: NDArray) -> float:
        """
        Run a single iteration of the EM algorithm max_iter times or til the
        difference in losses is smaller than tol.
        """
        n_samples, n_features = X.shape
        # np.random.seed(42)
        self.debug_amount_of_random_inits += 1
        init_index = self.random_state.random_integers(
            low=0, high=(n_samples - 1), size=(1,)
        ).squeeze()
        self.mean = X[init_index]
        variance = np.var(X, axis=0)
        variance += np.spacing(variance.max())
        self.covariance_matrix = np.diag(variance)
        self.pi = 1 - self.p_outlier_init
        loss_old = np.inf
        loss = 0.0
        gammas, loss, _ = self._e_step(X)
        for i in range(self.max_iter):
            self._m_step(X, gammas)
            gammas, loss, _ = self._e_step(X)
            loss_diff = loss - loss_old
            # here is the acutalization of debug params
            self.debug_series["mean"].append(self.mean)
            self.debug_series["covariance"].append(self.covariance_matrix)
            self.debug_series["pi"].append(self.pi)
            self.debug_series["loss"].append(loss)
            if 0 <= loss_diff < self.tol * np.abs(loss):
                break
            loss_old = loss
            # self._m_step(X, gammas)
            # # here is the acutalization of debug params
            # self.debug_series["mean"].append(self.mean)
            # self.debug_series["covariance"].append(self.covariance_matrix)
            # self.debug_series["pi"].append(self.pi)
        return loss

    def _m_step(self, X: NDArray, gammas: NDArray) -> None:
        """
        Maximization step.
        """
        n_samples, n_features = X.shape
        self.pi = float(np.mean(gammas))
        self.mean = gammas.dot(X) / (n_samples * self.pi)
        Xc = (X - self.mean) * np.sqrt(gammas[:, None])
        self.covariance_matrix = (Xc.T @ Xc) / (n_samples * self.pi) + np.spacing(
            self.covariance_matrix
        ) * np.eye(n_features)
        # self.debug_series["mean"].append(self.mean)
        # self.debug_series["covariance"].append(self.covariance_matrix)
        # self.debug_series["pi"].append(self.pi)

    def _cholesky(self, max_attempts: int = 5) -> NDArray:
        """
        Compute the Cholesky decomposition of the covariance matrix.
        """
        try:
            L = np.linalg.cholesky(self.covariance_matrix)
        except np.linalg.LinAlgError:  # covariance_matrix is not positive definite
            for i in range(5):  # try regularizing it several times
                w, v = sp.linalg.eigh(self.covariance_matrix)
                w = np.maximum(w, np.spacing(w.max()))
                self.covariance_matrix = v @ np.diag(w) @ v.T
                try:
                    L = np.linalg.cholesky(self.covariance_matrix)
                    break
                except np.linalg.LinAlgError:
                    continue
            else:  # if it still fails, raise an error
                raise np.linalg.LinAlgError
        return L

    def _get_nlogl(self, X: NDArray, debug_predict_flag=False) -> Tuple[float, NDArray]:
        """
        Get log likelihood of pristine class.
        """
        n_samples, n_features = X.shape
        L = self._cholesky()  # covariance_matrix = L@L.T
        D = np.diag(L)
        Xc = X - self.mean
        # Mahalanobis distance is now the L2 norm of L⁻¹ @ Xc.T
        # along the components axis
        Xc_m = np.linalg.solve(L, Xc.T)
        mahalanobis = np.sum(Xc_m**2, axis=0)
        self.debug_amount_of_fits += 1
        nlogl = 0.5 * (mahalanobis + n_features * np.log(2 * np.pi)) + np.sum(np.log(D))

        self.debug_series["nlogl"].append(nlogl)
        return nlogl, mahalanobis

    def _e_step(
        self, X: NDArray, debug_predict_flag=False
    ) -> Tuple[NDArray, float, NDArray]:
        """
        Run the expectation step.
        """
        nlogl, mahal = self._get_nlogl(X)
        log_gammas_inlier = np.log(self.pi) - nlogl
        log_gammas_outlier = np.log(1 - self.pi) - self.outlier_nlogl
        log_gammas_inlier = log_gammas_inlier[:, None]
        log_gammas_outlier = log_gammas_outlier.repeat(log_gammas_inlier.shape[0])[
            :, None
        ]
        log_gammas = np.append(log_gammas_inlier, log_gammas_outlier, axis=1)
        max_log_likelihood = np.max(log_gammas, axis=1, keepdims=True)
        gammas = np.exp(log_gammas - max_log_likelihood)
        dem = np.sum(gammas, axis=1, keepdims=True)
        gammas /= dem
        loss = np.mean(np.log(dem) + max_log_likelihood)
        # equivalent to a softmax but we also compute the loss
        # gammas = sp.special.softmax(log_gammas, axis=1)
        return gammas[:, 0], loss, mahal

    def predict(self, X: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Predict the class of the samples.
        """
        gammas, _, mahal = self._e_step(X)
        return gammas, mahal
