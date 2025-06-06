import os

import cv2 as cv
import mpmath
import numpy as np
from arrow import get
from networkx import k_components
from scipy.stats import binom
from sympy import true
from test_zs import true_zs

from photoholmes.utils.image import (
    image_to_luminance,
    plot,
    plot_multiple,
    read_image,
    save_image,
)

true_zs = np.loadtxt("data/debug/true_zs.csv", delimiter=",")

REPO_DIR = "/home/dsense/extra/tesis/photoholmes/extra/zero/zero"
IMAGES = ["tampered1.png", "tampered1.jpg", "tampered1_99.jpg"]
IMAGE = IMAGES[2]

NO_VOTE = -1


def check_outputs_coincide(given_output, true_im_name="DEBUG.png"):
    debug = read_image(os.path.join(REPO_DIR, true_im_name))
    assert np.allclose(given_output, debug)


def compute_grid_votes_per_pixel(luminance: np.ndarray) -> np.ndarray:
    """
    Compute the grid votes per pixel.
    :param luminance: Luminance image.
    :return: Grid votes per pixel.
    """
    X, Y = luminance.shape
    zeros = np.zeros_like(luminance, dtype=np.int32)
    votes = np.zeros_like(luminance, dtype=np.int32)

    zs = np.empty_like(luminance, dtype=np.int32)

    cos_t = np.cos(np.outer(2 * np.arange(8) + 1, np.arange(8)) * np.pi / 16)

    const_along_x = np.all(luminance[:, :, np.newaxis] == luminance[:, :1], axis=(1, 2))
    const_along_y = np.all(luminance[:, :, np.newaxis] == luminance[:1, :], axis=(1, 2))

    for x in range(X - 7):
        for y in range(Y - 7):
            z = 0

            for i in range(8):
                for j in range(8):
                    if i > 0 or j > 0:
                        dct_ij = (
                            luminance[y : y + 8, x : x + 8]
                            * np.outer(cos_t[:, j], cos_t[:, i])
                        ).sum() * (
                            0.25
                            * (1 / np.sqrt(2.0) if i == 0 else 1)
                            * (1 / np.sqrt(2.0) if j == 0 else 1)
                        )
                        if abs(dct_ij) < 0.5:
                            z += 1
            # dct_ij = np.sum(
            #     luminance[y : y + 8, x : x + 8] * cos_t[:, :, np.newaxis, np.newaxis],
            #     axis=(0, 1),
            # ) * (0.25 * (1 / np.sqrt(2.0)) * (1 / np.sqrt(2.0)))

            # mask = np.abs(dct_ij) < 0.5
            # z = np.sum(mask)

            zs[y, x] = z
            z = true_zs[x][y]  # DEBUG

            mask_zeros = z == zeros[y : y + 8, x : x + 8]
            mask_greater = z > zeros[y : y + 8, x : x + 8]

            votes[y : y + 8, x : x + 8][mask_zeros] = NO_VOTE
            zeros[y : y + 8, x : x + 8][mask_greater] = z
            votes[y : y + 8, x : x + 8][mask_greater] = (
                NO_VOTE
                if const_along_x[y] or const_along_y[x]
                else (x % 8) + (y % 8) * 8
            )

    votes[:7, :] = votes[-7:, :] = votes[:, :7] = votes[:, -7:] = NO_VOTE
    print("Cantidad de elementos con distancia >2:", (np.abs(zs.T - true_zs) > 2).sum())
    print("Maxima diferencia de zs:", (np.abs(zs.T - true_zs)).max())
    return votes


# def log_nfa(n, k, p, logNT):
#     inv = np.zeros(TABSIZE)
#     tolerance = 0.1
#     log1term, term, bin_term, mult_term, bin_tail, err = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
#     p_term = p / (1.0 - p)
#     if n < 0 or k < 0 or k > n or p < 0.0 or p > 1.0:
#         error("wrong n, k or p values in nfa()")
#     if n == 0 or k == 0:
#         return logNT
#     log1term = (
#         gammaln(n + 1.0)
#         - gammaln(k + 1.0)
#         - gammaln(n - k + 1.0)
#         + k * math.log(p)
#         + (n - k) * math.log(1.0 - p)
#     )
#     term = math.exp(log1term)
#     if term == 0.0:
#         if k > n * p:
#             return log1term / M_LN10 + logNT
#         else:
#             return logNT
#     bin_tail = term
#     for i in range(k + 1, n + 1):
#         bin_term = (n - i + 1) * (inv[i] if i < TABSIZE and inv[i] != 0 else (1.0 / i))
#         mult_term = bin_term * p_term
#         term *= mult_term
#         bin_tail += term
#         if bin_term < 1.0:
#             err = term * ((1.0 - pow(mult_term, (n - i + 1))) / (1.0 - mult_term) - 1.0)
#             if err < tolerance * abs(-math.log10(bin_tail) - logNT) * bin_tail:
#                 break
#     return math.log10(bin_tail) + logNT


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
        print("Numpy implementation")
        return np.log10(1 - cdf)
    else:
        bin_tail = np.empty_like(ks)
        for i, k in enumerate(ks):
            # bin_tail[i] = np.sum(
            #     np.array([bin_prob(x, n, p) for x in range(int(k), int(k) + 100)])
            # )
            bin_tail[i] = mpmath.nsum(
                lambda x: bin_prob(x, n, p), [int(k), int(k) + 50]
            )
        log_bin_tail = np.log10(bin_tail)

        return log_bin_tail


def log_nfa(N_tests, ks, n, p):
    return np.log10(N_tests) + log_bin_tail(ks, n, p)


def compare_arrays(true_array, estimated_array, threshold, array_name=""):
    print("Arrays cercanos") if np.allclose(
        true_array, estimated_array, atol=threshold
    ) else print("Arrays distintos")
    print(
        f"{array_name}: Cantidad de elementos con distancia >{threshold}:",
        (np.abs(true_array - estimated_array) > threshold).sum(),
    )
    print(
        f"{array_name}: Maxima diferencia:",
        (np.abs(true_array - estimated_array)).max(),
    )
    print(true_array, estimated_array, true_array - estimated_array)


def detect_global_grids(votes):
    X, Y = votes.shape
    grid_votes = np.zeros(64)
    max_votes = 0
    most_voted_grid = -1
    p = 1.0 / 64.0
    for x in range(X):
        for y in range(Y):
            if votes[y, x] >= 0 and votes[y, x] < 64:
                grid = votes[y, x]
                grid_votes[grid] += 1
                if grid_votes[grid] > max_votes:
                    max_votes = grid_votes[grid]
                    most_voted_grid = grid

    true_lnfa_grids = np.loadtxt("data/debug/true_lnfa_grids.csv", delimiter=",")
    N_tests = (64 * X * Y) ** 2
    ks = np.floor(grid_votes / 64) - 1
    n = np.ceil(X * Y / 64)
    p = 1 / 64
    lnfa_grids = log_nfa(N_tests, ks, n, p)
    compare_arrays(true_lnfa_grids, lnfa_grids, 1e-1, "lnfa_grids")

    grid_meaningful = (
        most_voted_grid >= 0
        and most_voted_grid < 64
        and lnfa_grids[most_voted_grid] < 0.0
    )

    return most_voted_grid if grid_meaningful else -1


im = read_image(os.path.join(REPO_DIR, IMAGE))
luminance = image_to_luminance(im)
votes = compute_grid_votes_per_pixel(luminance)
main_grid = detect_global_grids(votes)
