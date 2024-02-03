import os

import cv2 as cv
import numpy as np
from arrow import get
from scipy.fftpack import dctn
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
IMAGE = IMAGES[0]

NO_VOTE = -1


def check_outputs_coincide(given_output, true_im_name="DEBUG.png"):
    debug = read_image(os.path.join(REPO_DIR, true_im_name))
    assert np.allclose(given_output, debug)


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
            # z = 0
            # for i in range(8):
            #     for j in range(8):
            #         if i > 0 or j > 0:
            #             dct_ij = (
            #                 luminance[y : y + 8, x : x + 8]
            #                 * np.outer(cos_t[:, j], cos_t[:, i])
            #             ).sum() * (
            #                 0.25
            #                 * (1 / np.sqrt(2.0) if i == 0 else 1)
            #                 * (1 / np.sqrt(2.0) if j == 0 else 1)
            #             )
            #             if abs(dct_ij) < 0.5:
            #                 z += 1
            dct = dctn(luminance[y : y + 8, x : x + 8], type=2, norm="ortho")
            z = (np.abs(dct) < 0.5).sum()
            border_cases += ((np.abs(dct) < 0.501) & (np.abs(dct) > 0.499)).sum()

            zs[y, x] = z
            # z = true_zs[x][y]  # DEBUG

            mask_zeros = z == zeros[y : y + 8, x : x + 8]
            mask_greater = z > zeros[y : y + 8, x : x + 8]

            votes[y : y + 8, x : x + 8][mask_zeros] = NO_VOTE
            zeros[y : y + 8, x : x + 8][mask_greater] = z
            votes[y : y + 8, x : x + 8][mask_greater] = (
                NO_VOTE
                if const_along_x[y] or const_along_y[x]
                else (x % 8) + (y % 8) * 8
            )
    print("Border cases:", border_cases / (64 * X * Y))
    votes[:7, :] = votes[-7:, :] = votes[:, :7] = votes[:, -7:] = NO_VOTE
    print(
        "Cantidad de elementos con distancia >0:",
        (np.abs(zs.T - true_zs) > 0).mean(),
    )
    print("Cantidad de elementos con distancia >2:", (np.abs(zs.T - true_zs) > 2).sum())
    print("Maxima diferencia de zs:", (np.abs(zs.T - true_zs)).max())
    return votes


im = read_image(os.path.join(REPO_DIR, IMAGE))
luminance = image_to_luminance(im)
votes = compute_grid_votes_per_pixel(luminance)
true_votes = np.loadtxt("data/debug/true_votes.csv", delimiter=",")

# cv.imwrite(os.path.join("data/debug", "DEBUG.png"), votes)
# votes = read_image(os.path.join("data/debug", "DEBUG.png"))
# true_votes = read_image(os.path.join(REPO_DIR, "DEBUG.png"))

print("Los votos coinciden:", (true_votes == votes).all())
plot_multiple(
    [
        true_votes,
        votes,
        true_votes - votes,
    ]
)
# check_outputs_coincide(votes)
