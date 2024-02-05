import mpmath
import numpy as np
from numpy.typing import NDArray
from scipy.stats import binom


def bin_prob(k: int, n: int, p: float) -> float:
    """
    P(X = k) where X~ Bin(n, p).
    Input: k, n, p parameters of Binomial Tail
    Output: P(X = k)
    """
    arr = mpmath.binomial(n, k)
    pk = mpmath.power(p, k)
    pp = mpmath.power(1 - p, n - k)
    aux = mpmath.fmul(pk, pp)
    bp = mpmath.fmul(arr, aux)
    return bp


def binom_tail(ks: np.ndarray, n: int, p: float) -> NDArray:
    """
    P(X >= k) where X~ Bin(n, p), for each k in ks.
    Input:
     - ks: array of k values.
     - n: total amount of independent Bernoulli experiments.
     - p: probability of success of each Bernoulli experiment.
    Output:
     - array of P(X >= k) for each k in ks.
    """
    cdf = binom.cdf(ks, n, p)
    if (cdf != 1).all():
        print("Here")
        return 1 - cdf
    else:
        cdf = np.zeros_like(ks)
        for i, k in enumerate(ks):
            cdf[i] = np.sum(np.array([bin_prob(x, n, p) for x in range(int(k))]))
        cdf[cdf > 1] = 1
        return 1 - cdf


def log_bin_tail(ks: NDArray, n: int, p: float) -> NDArray:
    """
    Computes the array of the logarithm of the binomial tail, for an array of k values,
    and two fixed parameters n,p. Computes a light or high-precision version as needed.
    """
    cdf = binom.cdf(ks, n, p)
    if (cdf != 1).all():
        return np.log10(1 - cdf)
    else:
        bin_tail = np.empty_like(ks)
        for i, k in enumerate(ks):
            bin_tail[i] = mpmath.nsum(
                lambda x: bin_prob(x, n, p), [int(k), int(k) + 50]
            )
        log_bin_tail = np.log10(bin_tail)

        return log_bin_tail


def log_nfa(N_tests: int, ks: NDArray, n: int, p: float) -> NDArray:
    """
    Computes the array of the logarithm of NFA for a given amount N_tests,
     an array of k values, and two fixed parameters n,p.
    """
    return np.log10(N_tests) + log_bin_tail(ks, n, p)
