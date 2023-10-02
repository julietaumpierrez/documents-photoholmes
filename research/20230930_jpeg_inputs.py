# %%
import os
import time
from tempfile import NamedTemporaryFile

import cv2 as cv
import jpegio
import matplotlib.pyplot as plt
import numpy as np
from PIL.Image import open
from scipy.fftpack import dct, dctn, idct
from scipy.interpolate import interp2d
from scipy.ndimage import zoom

from photoholmes.models.method_factory import MethodFactory
from photoholmes.utils import image

# os.chdir("..")

# %%
DATA_DIR = "benchmarking/test_images/"

IMAGES_PATH = DATA_DIR + "images/"
MASK_PATH = DATA_DIR + "masks/"

image_path = IMAGES_PATH + "Im_3.jpg"

img = open(image_path)
image.plot_multiple_images([img], ncols=2)

# %%
image_names = [
    path
    for path in os.listdir(IMAGES_PATH)
    if ((path[-4:] == ".jpg") or (path[-4:] == ".JPG"))
]
for image_name in image_names:
    image_path = IMAGES_PATH + image_name
    jpeg = jpegio.read(str(image_path))
    num_channels = len(jpeg.coef_arrays)
    ci = jpeg.comp_info

    sampling_factors = np.array(
        [[ci[i].v_samp_factor, ci[i].h_samp_factor] for i in range(num_channels)]
    )
    if num_channels == 3:
        if (sampling_factors[:, 0] == sampling_factors[0, 0]).all():
            sampling_factors[:, 0] = 2
        if (sampling_factors[:, 1] == sampling_factors[0, 1]).all():
            sampling_factors[:, 1] = 2
    else:
        sampling_factors[:, :] = 2

    DCT_coef = np.empty((num_channels, *jpeg.coef_arrays[0].shape))

    for i in range(num_channels):
        r, c = jpeg.coef_arrays[i].shape
        block_coefs = (
            jpeg.coef_arrays[i].reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3)
        )
        r_factor, c_factor = 2 // sampling_factors[i][0], 2 // sampling_factors[i][1]
        channel_coefficients = np.zeros((r * r_factor, c * c_factor))
        channel_coefficient_blocks = channel_coefficients.reshape(
            r // 8, r_factor * 8, c // 8, c_factor * 8
        ).transpose(0, 2, 1, 3)
        channel_coefficient_blocks[:, :, :, :] = np.tile(
            block_coefs, (r_factor, c_factor)
        )

        DCT_coef[i] = channel_coefficients

    assert (DCT_coef == image.read_as_jpeg(image_path)).all()

# %%
"""Deprecated function:"""


def _calculate_DCT(image_path: str) -> np.ndarray:
    # FIXME: Charlar con tutores acerca de esto vs. descargar como jpeg como hacen en CATNet
    """Computes the DCT coefficients of a given image."""
    image = np.array(open(image_path).convert("YCbCr"))

    r, c, nc = image.shape
    image_blocks = non_overlapping_blocks(image)

    dct_coeffs_blocks = dctn(image_blocks, type=2, norm="ortho", axes=(-2, -1))

    dct_coeffs = np.array(
        [
            [
                [
                    dct_coeffs_blocks[i // 8, j // 8, i % 8, j % 8, channel]
                    for j in range(c)
                ]
                for i in range(r)
            ]
            for channel in range(nc)
        ]
    ).round()

    return dct_coeffs


def non_overlapping_blocks(
    image: np.ndarray, block_shape: tuple = (8, 8)
) -> np.ndarray:
    """Divide an image into non-overlapping blocks of shape "block_shape".
    TODO: Add border cases when image shape is not divisible by block shape."""
    h, w, nc = image.shape
    bh, bw = block_shape
    strides = (w, 1, w, 1)
    blocked_image_shape = (h // bh, w // bw, bh, bw)
    blocks = np.empty((*blocked_image_shape, 3))
    print(blocks.shape)
    for channel in range(nc):
        blocks[:, :, :, :, channel] = np.lib.stride_tricks.as_strided(
            image[:, :, channel], shape=blocked_image_shape, strides=strides
        )
    return blocks


# %%
image_path = IMAGES_PATH + "Im_3.jpg"

# Method 1: Already jpeg
t0 = time.time()
DCT_coef = image._read_jpeg(IMAGES_PATH + "Im_3.jpg")

# Method 2: saved jpeg and reopened
t1 = time.time()
temp = NamedTemporaryFile(suffix=".jpg")
open(image_path).convert("RGB").save(temp.name, quality=100, subsampling=0)
t11 = time.time()
DCT_coef_reconverted = image._read_jpeg(temp.name)

# Method 3: Calculated DCTs
t2 = time.time()
temp2 = NamedTemporaryFile(suffix=".png")
open(image_path).convert("RGB").save(temp2.name)
DCT_coef_tform = _calculate_DCT(temp2.name)
t3 = time.time()


print(DCT_coef[0, :8, :8])
print("Elapsed time:", t1 - t0)
print(DCT_coef_reconverted[0, :8, :8])
print("Elapsed time:", t2 - t1, ", Jpeg pipeline:", t11 - t1)
print(DCT_coef_tform[0, :8, :8])
print("Elapsed time:", t3 - t2)

# No son iguales a menos del factor de matriz de cuantización. Esto no afecta el efecto DQ.
# %%
for image_name in os.listdir(IMAGES_PATH):
    image_path = IMAGES_PATH + image_name
    DCT_coef = image.read_as_jpeg(image_path)
