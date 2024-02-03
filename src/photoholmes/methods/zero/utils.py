import mpmath
import numpy as np
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


def binom_tail(ks: np.ndarray, n: int, p: float) -> np.ndarray:
    """
    TODO: update docstring
    TODO: integrate in general utils and noisesniffer
    P(x >= np.floor(K/w**2)) where X~ Bin(np.ceil(N/w**2), m). If the precision
    of scipy is not high enough, the computation is done using mpmath library
    (see bin_prob function)
    Input: K, N, w, m parameters of Binomial Tail according to the NFA formula of the
    paper
    Output: Binomial Tail
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


def log_bin_tail(ks, n, p):
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


def log_nfa(N_tests, ks, n, p):
    return np.log10(N_tests) + log_bin_tail(ks, n, p)
