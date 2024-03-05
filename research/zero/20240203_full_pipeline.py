import os
from typing import Dict, List, Tuple

import cv2 as cv
import mpmath
import numpy as np
from scipy.fftpack import dctn
from scipy.stats import binom
from torch import Tensor

from photoholmes.methods.base import BaseMethod
from photoholmes.methods.DQ.utils import ZIGZAG, fft_period, histogram_period
from photoholmes.postprocessing.resizing import (
    resize_heatmap_with_trim_and_pad,
    simple_upscale_heatmap,
)
from photoholmes.utils.image import (
    plot,
    plot_multiple,
    read_image,
    save_image,
    tensor2numpy,
)

# true_zs = np.loadtxt("data/debug/true_zs.csv", delimiter=",")
# true_votes = np.loadtxt("data/debug/true_votes.csv", delimiter=",")

REPO_DIR = "/home/dsense/extra/tesis/photoholmes/extra/zero/zero"
IMAGES = ["tampered1.png", "tampered1.jpg", "tampered1_99.jpg"]
IMAGE = IMAGES[0]

NO_VOTE = -1

# class Zero(BaseMethod):
#     def __init__(self, **kwargs) -> None:
#         """
#         Initialize the DQ class.

#         :param number_frecs: Number of frequencies, defaults to 10.
#         :param kwargs: Additional keyword arguments.
#         """
#         super().__init__(**kwargs)

#     def predict(
#         self, dct_coefficients: NDArray, original_image_size: Tuple[int, int]
#     ) -> Dict[str, Tensor]:


def image_to_luminance(image: Tensor) -> np.ndarray:
    """Converts an image to luminance."""
    np_image = tensor2numpy(image)
    return cv.cvtColor(np_image, cv.COLOR_RGB2GRAY)


# Auxiliary functions
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

    border_cases = 0
    cos_t = np.cos(np.outer(2 * np.arange(8) + 1, np.arange(8)) * np.pi / 16)

    const_along_x = np.all(luminance[:, :, np.newaxis] == luminance[:, :1], axis=(1, 2))
    const_along_y = np.all(luminance[:, :, np.newaxis] == luminance[:1, :], axis=(1, 2))

    for x in range(X - 7):
        for y in range(Y - 7):
            dct = dctn(luminance[y : y + 8, x : x + 8], type=2, norm="ortho")
            z = (np.abs(dct) < 0.5).sum()

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

    return votes


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


def compare_arrays(true_array, estimated_array, threshold, array_name=""):
    print(
        f"{array_name}: Cantidad de elementos con distancia >{threshold}:",
        (np.abs(true_array - estimated_array) > threshold).sum(),
    )
    print(
        f"{array_name}: Maxima diferencia:",
        (np.abs(true_array - estimated_array)).max(),
    )
    print(true_array, estimated_array, true_array - estimated_array)
    (
        print("Arrays cercanos")
        if np.allclose(true_array, estimated_array, atol=threshold)
        else print("Arrays distintos")
    )


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
    N_tests = (64 * X * Y) ** 2
    ks = np.floor(grid_votes / 64) - 1
    n = np.ceil(X * Y / 64)
    p = 1 / 64
    lnfa_grids = log_nfa(N_tests, ks, n, p)

    grid_meaningful = (
        most_voted_grid >= 0
        and most_voted_grid < 64
        and lnfa_grids[most_voted_grid] < 0.0
    )

    return most_voted_grid if grid_meaningful else NO_VOTE


def detect_forgeries(votes, grid_to_exclude):
    grid_max = 63
    p = 1.0 / 64.0
    X, Y = votes.shape
    N_tests = (64 * X * Y) ** 2
    mask_aux = np.zeros_like(votes, dtype=int)
    used = np.full_like(votes, False)
    reg_x = np.zeros(votes.shape[0] * votes.shape[1], dtype=int)
    reg_y = np.zeros(votes.shape[0] * votes.shape[1], dtype=int)
    foreign_regions, forgery_mask, forgery_mask_reg = (
        np.empty_like(votes),
        np.empty_like(votes),
        np.empty_like(votes),
    )
    W = 9
    min_size = np.ceil(64.0 * np.log10(N_tests) / np.log10(64.0)).astype(int)

    for x in range(X):
        for y in range(Y):
            if (
                not used[y, x]
                and votes[y, x] != grid_to_exclude
                and votes[y, x] >= 0
                and votes[y, x] <= grid_max
            ):
                grid = votes[y, x]
                corner_0 = corner_1 = np.array([x, y])
                used[y, x] = True
                reg_x[0] = x
                reg_y[0] = y
                reg_size = 1
                i = 0

                while i < reg_size:
                    lower_xx = max(reg_x[i] - W, 0)
                    higher_xx = min(reg_x[i] + W + 1, X)
                    lower_yy = max(reg_y[i] - W, 0)
                    higher_yy = min(reg_y[i] + W + 1, Y)
                    for xx in range(lower_xx, higher_xx):
                        for yy in range(lower_yy, higher_yy):
                            if not used[yy, xx] and votes[yy, xx] == grid:
                                used[yy, xx] = True
                                reg_x[reg_size] = xx
                                reg_y[reg_size] = yy
                                reg_size += 1

                                corner_0 = np.min(
                                    np.vstack([corner_0, np.array([xx, yy])]),
                                    axis=1,
                                )
                                corner_1 = np.max(
                                    np.vstack([corner_0, np.array([xx, yy])]),
                                    axis=1,
                                )
                    i += 1
                if reg_size >= min_size:
                    n = int(
                        (corner_1[0] - corner_0[0] + 1)
                        * (corner_1[1] - corner_1[1] + 1)
                        // 64
                    )
                    k = int(reg_size // 64)
                    lnfa = log_nfa(N_tests, np.array([k]), n, p)[0]
                    if lnfa < 0.0:
                        idxs = np.array([reg_x[:reg_size], reg_y[:reg_size]]).T
                        forgery_mask[idxs[:, 1], idxs[:, 0]] = 255

    return forgery_mask


# Auxiliary functions
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

    border_cases = 0
    cos_t = np.cos(np.outer(2 * np.arange(8) + 1, np.arange(8)) * np.pi / 16)

    const_along_x = np.all(luminance[:, :, np.newaxis] == luminance[:, :1], axis=(1, 2))
    const_along_y = np.all(luminance[:, :, np.newaxis] == luminance[:1, :], axis=(1, 2))

    for x in range(X - 7):
        for y in range(Y - 7):
            dct = dctn(luminance[y : y + 8, x : x + 8], type=2, norm="ortho")
            z = (np.abs(dct) < 0.5).sum()

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

    return votes


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


def compare_arrays(true_array, estimated_array, threshold, array_name=""):
    print(
        f"{array_name}: Cantidad de elementos con distancia >{threshold}:",
        (np.abs(true_array - estimated_array) > threshold).sum(),
    )
    print(
        f"{array_name}: Maxima diferencia:",
        (np.abs(true_array - estimated_array)).max(),
    )
    print(true_array, estimated_array, true_array - estimated_array)
    (
        print("Arrays cercanos")
        if np.allclose(true_array, estimated_array, atol=threshold)
        else print("Arrays distintos")
    )


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
    N_tests = (64 * X * Y) ** 2
    ks = np.floor(grid_votes / 64) - 1
    n = np.ceil(X * Y / 64)
    p = 1 / 64
    lnfa_grids = log_nfa(N_tests, ks, n, p)

    grid_meaningful = (
        most_voted_grid >= 0
        and most_voted_grid < 64
        and lnfa_grids[most_voted_grid] < 0.0
    )

    return most_voted_grid if grid_meaningful else NO_VOTE


def detect_forgeries(votes, grid_to_exclude):
    grid_max = 63
    p = 1.0 / 64.0
    X, Y = votes.shape
    N_tests = (64 * X * Y) ** 2
    mask_aux = np.zeros_like(votes, dtype=int)
    used = np.full_like(votes, False)
    reg_x = np.zeros(votes.shape[0] * votes.shape[1], dtype=int)
    reg_y = np.zeros(votes.shape[0] * votes.shape[1], dtype=int)
    foreign_regions, forgery_mask, forgery_mask_reg = (
        np.empty_like(votes),
        np.empty_like(votes),
        np.empty_like(votes),
    )
    W = 9
    min_size = np.ceil(64.0 * np.log10(N_tests) / np.log10(64.0)).astype(int)

    for x in range(X):
        for y in range(Y):
            if (
                not used[y, x]
                and votes[y, x] != grid_to_exclude
                and votes[y, x] >= 0
                and votes[y, x] <= grid_max
            ):
                grid = votes[y, x]
                corner_0 = corner_1 = np.array([x, y])
                used[y, x] = True
                reg_x[0] = x
                reg_y[0] = y
                reg_size = 1
                i = 0

                while i < reg_size:
                    lower_xx = max(reg_x[i] - W, 0)
                    higher_xx = min(reg_x[i] + W + 1, X)
                    lower_yy = max(reg_y[i] - W, 0)
                    higher_yy = min(reg_y[i] + W + 1, Y)
                    for xx in range(lower_xx, higher_xx):
                        for yy in range(lower_yy, higher_yy):
                            if not used[yy, xx] and votes[yy, xx] == grid:
                                used[yy, xx] = True
                                reg_x[reg_size] = xx
                                reg_y[reg_size] = yy
                                reg_size += 1

                                corner_0 = np.min(
                                    np.vstack([corner_0, np.array([xx, yy])]),
                                    axis=1,
                                )
                                corner_1 = np.max(
                                    np.vstack([corner_0, np.array([xx, yy])]),
                                    axis=1,
                                )
                    i += 1
                if reg_size >= min_size:
                    n = int(
                        (corner_1[0] - corner_0[0] + 1)
                        * (corner_1[1] - corner_1[1] + 1)
                        // 64
                    )
                    k = int(reg_size // 64)
                    lnfa = log_nfa(N_tests, np.array([k]), n, p)[0]
                    if lnfa < 0.0:
                        idxs = np.array([reg_x[:reg_size], reg_y[:reg_size]]).T
                        forgery_mask[idxs[:, 1], idxs[:, 0]] = 255

    return forgery_mask


im = read_image(os.path.join(REPO_DIR, IMAGE))
luminance = image_to_luminance(im)
votes = compute_grid_votes_per_pixel(luminance)
main_grid = detect_global_grids(votes)
forgery_mask = detect_forgeries(votes, main_grid)

true_forgery_mask = np.loadtxt("data/debug/true_forgery_mask.csv", delimiter=",")
compare_arrays(forgery_mask, true_forgery_mask, 1, "forgery_mask")
plot_multiple([votes, forgery_mask, true_forgery_mask])
