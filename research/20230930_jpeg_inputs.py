# %%
import os

import cv2 as cv
import jpegio
import matplotlib.pyplot as plt
import numpy as np
from PIL.Image import open
from scipy.fftpack import dct, idct
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
