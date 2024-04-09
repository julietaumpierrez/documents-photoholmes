# %%
import os

if "research" in os.getcwd():
    os.chdir("../..")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")


# %%
import numpy as np
from PIL import Image

votes_orig = np.array(Image.open("../implementations/zero/test/votes.tif"))
zeros_orig = np.array(Image.open("../implementations/zero/test/zeros.tif"))
votes_orig, zeros_orig
# %%
from numpy.typing import NDArray
from scipy.fftpack import dctn


def compute_grid_votes_per_pixel(luminance: NDArray) -> NDArray:
    """
    Compute the grid votes per pixel.

    Args:
        luminance (NDArray): Input luminance channel.

    Returns:
        NDArray: Grid votes per pixel.
    """
    Y, X = luminance.shape
    zeros = np.zeros_like(luminance, dtype=np.int16)
    votes = np.empty_like(luminance, dtype=np.int16)
    votes[:] = -1

    for x in range(X - 7):
        for y in range(Y - 7):
            block = luminance[y : y + 8, x : x + 8]
            const_along_x = np.all(block[:, :] == block[:1, :])
            const_along_y = np.all(block[:, :] == block[:, :1])

            dct = dctn(block, type=2, norm="ortho")
            dct[0, 0] = 1  # discard constant component

            z = (np.abs(dct) < 0.5).sum()

            mask_tie = z == zeros[y : y + 8, x : x + 8]
            votes[y : y + 8, x : x + 8][mask_tie] = -1

            mask_greater = z > zeros[y : y + 8, x : x + 8]
            zeros[y : y + 8, x : x + 8][mask_greater] = z
            if const_along_x or const_along_y:
                votes[y : y + 8, x : x + 8][mask_greater] = -1
            else:
                votes[y : y + 8, x : x + 8][mask_greater] = (x % 8) + (y % 8) * 8

            if (
                x > 7
                and y > 7
                and (not const_along_x)
                and (not const_along_y)
                and votes[y, x] != votes_orig[y, x]
            ):
                print(x, y, votes[y, x], votes_orig[y, x])

    votes[:7, :] = votes[-7:, :] = votes[:, :7] = votes[:, -7:] = -1

    return votes, zeros


# %%
import matplotlib.pyplot as plt

from photoholmes.methods.zero.preprocessing import zero_preprocessing
from photoholmes.utils.image import read_image

img = read_image("data/COVERAGE/image/21t.tif")
img = zero_preprocessing(image=img)["image"][:, :, 0]
plt.imshow(img, cmap="gray")
# %%
from time import time

t0 = time()
votes, zeros = compute_grid_votes_per_pixel(img)
print(time() - t0)

# %%
x = 6
y = 200
votes[y : y + 5, x : x + 5], votes_orig[y : y + 5, x : x + 5]
# %%
zeros[y : y + 5, x : x + 5], zeros_orig[y : y + 5, x : x + 5]
# %%
diff = votes != votes_orig
print("Diff count {0}".format(diff.sum()), "Total count {0}".format(votes_orig.size))
print(f"Diff -1: {(votes[diff] == -1).sum() + (votes_orig[diff] == -1).sum()}")

# %%
from photoholmes.methods.zero import Zero

zero = Zero()
# %%
main_grid = zero.detect_global_grids(votes_orig)
print(main_grid)
# %%
main_grid = zero.detect_global_grids(votes)
print(main_grid)
# %%
forgery_mask = zero.detect_forgeries(votes_orig, main_grid)
# %%
plt.imshow(forgery_mask, cmap="gray")

# %%
import time

t0 = time.time()
for x in range(1000):
    for y in range(1000):
        pass
tf = time.time()
print(tf - t0)
# %%
t0 = time.time()
tuples_list = [(i, j) for i in range(1000 + 1) for j in range(1000 + 1)]
for i, j in tuples_list:
    pass
tf = time.time()
print(tf - t0)


# %%
def compute_grid_votes_per_pixel(luminance):
    """
    Compute the grid votes per pixel.

    Args:
        luminance (NDArray): Input luminance channel.

    Returns:
        NDArray: Grid votes per pixel.
    """
    Y, X = luminance.shape
    zeros = np.zeros_like(luminance, dtype=np.int32)
    votes = np.full_like(luminance, -1, dtype=np.int32)

    for x in range(X - 7):
        for y in range(Y - 7):
            block = luminance[y : y + 8, x : x + 8]
            const_along_x = np.all(block[:, :] == block[:1, :])
            const_along_y = np.all(block[:, :] == block[:, :1])

            dct = dctn(block, type=2, norm="ortho")
            dct[0, 0] = 1  # Discard DC component
            z = (np.abs(dct) < 0.5).sum()

            mask_zeros = z == zeros[y : y + 8, x : x + 8]
            mask_greater = z > zeros[y : y + 8, x : x + 8]

            votes[y : y + 8, x : x + 8][mask_zeros] = -1
            zeros[y : y + 8, x : x + 8][mask_greater] = z
            votes[y : y + 8, x : x + 8][mask_greater] = (
                -1 if const_along_x or const_along_y else (x % 8) + (y % 8) * 8
            )

    votes[:7, :] = votes[-7:, :] = votes[:, :7] = votes[:, -7:] = -1

    return votes


t0 = time.time()
votes = compute_grid_votes_per_pixel(img)
print(time.time() - t0)


# %%
def compute_grid_votes_per_pixel(luminance):
    """
    Compute the grid votes per pixel.

    Args:
        luminance (NDArray): Input luminance channel.

    Returns:
        NDArray: Grid votes per pixel.
    """
    Y, X = luminance.shape
    zeros = np.zeros_like(luminance, dtype=np.int32)
    votes = np.full_like(luminance, -1, dtype=np.int32)

    for x in range(X - 7):
        for y in range(Y - 7):
            block = luminance[y : y + 8, x : x + 8]
            const_along_x = np.all(block[:, :] == block[:1, :])
            const_along_y = np.all(block[:, :] == block[:, :1])

            dct = dctn(block, type=2, norm="ortho")
            dct[0, 0] = 1  # Discard DC component
            z = (np.abs(dct) < 0.5).sum()

            zeros_block = zeros[y : y + 8, x : x + 8]
            votes_block = votes[y : y + 8, x : x + 8]

            mask_zeros = z == zeros_block
            votes_block[mask_zeros] = -1
            mask_greater = z > zeros_block

            zeros_block[mask_greater] = z
            votes_block[mask_greater] = (
                -1 if const_along_x or const_along_y else (x % 8) + (y % 8) * 8
            )

    votes[:7, :] = votes[-7:, :] = votes[:, :7] = votes[:, -7:] = -1

    return votes


print("Fast?")
t0 = time.time()
votes = compute_grid_votes_per_pixel(img)
print(time.time() - t0)
# %%
from photoholmes.methods.zero.utils import log_nfa


def detect_global_grids(votes: NDArray) -> int:
    """
    Detects the main estimated grid.

    Args:
        votes (NDArray): Grid votes per pixel. Each pixel votes 1 of the 64
            possible grids.

    Returns:
        int: Main detected grid.
    """
    Y, X = votes.shape
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

    print(grid_votes)
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

    return most_voted_grid if grid_meaningful else -1


t0 = time.time()
main_grid = detect_global_grids(votes)
print(time.time() - t0)
# %%
valid_votes = np.argwhere((votes >= 0) * (votes < 64))
valid_votes.max()
# %%
grid_votes, _ = np.histogram(
    votes[valid_votes[:, 0], valid_votes[:, 1]], bins=np.arange(65)
)
grid_votes


# %%
def detect_global_grids(votes: NDArray) -> int:
    """
    Detects the main estimated grid.

    Args:
        votes (NDArray): Grid votes per pixel. Each pixel votes 1 of the 64
            possible grids.

    Returns:
        int: Main detected grid.
    """
    Y, X = votes.shape
    grid_votes = np.zeros(64)
    p = 1.0 / 64.0

    valid_votes = np.argwhere((votes >= 0) * (votes < 64))
    grid_votes, _ = np.histogram(
        votes[valid_votes[:, 0], valid_votes[:, 1]], bins=np.arange(65)
    )
    most_voted_grid = int(np.argmax(grid_votes))

    N_tests = (64 * X * Y) ** 2

    ks = np.flor(grid_votes / 64) - 1
    n = np.ceil(X * Y / 64)
    p = 1 / 64
    lnfa_grids = log_nfa(N_tests, ks, n, p)

    grid_meaningful = (
        most_voted_grid >= 0
        and most_voted_grid < 64
        and lnfa_grids[most_voted_grid] < 0.0
    )

    return most_voted_grid if grid_meaningful else -1


t0 = time.time()
main_grid = detect_global_grids(votes)
print(time.time() - t0)

# %%
import mpmath
import numpy as np
from numpy.typing import NDArray
from scipy.stats import binom


def bin_prob(k: int, n: int, p: float) -> float:
    """
    Computes the binomial probability P(X = k) where X~ Bin(n, p).

    Args:
        k (int): number of successes.
        n (int): number of trials.
        p (float): probability of success.

    Returns:
        float: binomial probability.
    """
    arr = mpmath.binomial(n, k)
    pk = mpmath.power(p, k)
    pp = mpmath.power(1 - p, n - k)
    aux = mpmath.fmul(pk, pp)
    bp = mpmath.fmul(arr, aux)
    return bp


def binom_tail(ks: np.ndarray, n: int, p: float) -> NDArray:
    """
    Computes P(X >= k) where X~ Bin(n, p), for each k in ks.

    Args:
        ks (np.ndarray): array of k values.
        n (int): total amount of independent Bernoulli experiments.
        p (float): probability of success of each Bernoulli experiment.

    Returns:
        NDArray: array of P(X >= k) for each k in ks.
    """
    cdf = binom.cdf(ks, n, p)
    if (cdf != 1).all():
        return 1 - cdf
    else:
        cdf = np.zeros_like(ks)
        for i, k in enumerate(ks):
            cdf[i] = np.sum(np.array([bin_prob(x, n, p) for x in range(int(k))]))
        cdf[cdf > 1] = 1
        return 1 - cdf


# %%
ks = np.arange(64)
t0 = time.time()
