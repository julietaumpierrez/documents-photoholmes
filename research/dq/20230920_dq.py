# %%
import os

import cv2 as cv
import jpegio
import matplotlib.pyplot as plt
import numpy as np
from PIL.Image import open
from scipy.fftpack import dctn

from photoholmes.models.method_factory import MethodFactory
from photoholmes.utils import image

# os.chdir("..")

DATA_DIR = "benchmarking/test_images/"

IMAGES_PATH = DATA_DIR + "images/"
MASK_PATH = DATA_DIR + "masks/"

# %%
im_name = "Im_2.png"
os.listdir()
im_read = open(IMAGES_PATH + im_name)
im_read.save(IMAGES_PATH + im_name[:-4] + ".jpg")
image.plot_multiple_images([im_read], im_name, ncols=2)


# %%
def read_as_jpeg(image_path: str) -> np.ndarray:
    if image_path[-4:] == ".jpg" or image_path[-5:] == ".jpeg":
        return _read_jpeg(image_path)
    else:
        return _calculate_DCT(image_path)


def _read_jpeg(im_path):
    """
    :param im_path: JPEG image path
    :return: DCT_coef (Y,Cb,Cr), qtables (Y,Cb,Cr)
    """
    assert im_path[-4:] == ".jpg" or im_path[-5:] == ".jpeg"

    jpeg = jpegio.read(str(im_path))
    num_channels = len(jpeg.coef_arrays)

    # determine which axes to up-sample
    # ci = jpeg.
    ci = jpeg.comp_info
    need_scale = [
        [ci[i].v_samp_factor, ci[i].h_samp_factor] for i in range(num_channels)
    ]
    if num_channels == 3:
        if ci[0].v_samp_factor == ci[1].v_samp_factor == ci[2].v_samp_factor:
            need_scale[0][0] = need_scale[1][0] = need_scale[2][0] = 2
        if ci[0].h_samp_factor == ci[1].h_samp_factor == ci[2].h_samp_factor:
            need_scale[0][1] = need_scale[1][1] = need_scale[2][1] = 2
    else:
        need_scale[0][0] = 2
        need_scale[0][1] = 2

    # up-sample DCT coefficients to match image size
    DCT_coef = []
    DCT_coef = np.empty((num_channels, *jpeg.coef_arrays[0].shape))

    for i in range(num_channels):
        r, c = jpeg.coef_arrays[i].shape
        coef_view = (
            jpeg.coef_arrays[i].reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3)
        )
        # case 1: row scale (O) and col scale (O)
        if need_scale[i][0] == 1 and need_scale[i][1] == 1:
            out_arr = np.zeros((r * 2, c * 2))
            out_view = out_arr.reshape(r * 2 // 8, 8, c * 2 // 8, 8).transpose(
                0, 2, 1, 3
            )
            out_view[::2, ::2, :, :] = coef_view[:, :, :, :]
            out_view[1::2, ::2, :, :] = coef_view[:, :, :, :]
            out_view[::2, 1::2, :, :] = coef_view[:, :, :, :]
            out_view[1::2, 1::2, :, :] = coef_view[:, :, :, :]

        # case 2: row scale (O) and col scale (X)
        elif need_scale[i][0] == 1 and need_scale[i][1] == 2:
            out_arr = np.zeros((r * 2, c))
            # DCT_coef.append(out_arr)
            DCT_coef[i] = out_arr
            out_view = out_arr.reshape(r * 2 // 8, 8, c // 8, 8).transpose(0, 2, 1, 3)
            out_view[::2, :, :, :] = coef_view[:, :, :, :]
            out_view[1::2, :, :, :] = coef_view[:, :, :, :]

        # case 3: row scale (X) and col scale (O)
        elif need_scale[i][0] == 2 and need_scale[i][1] == 1:
            out_arr = np.zeros((r, c * 2))
            out_view = out_arr.reshape(r // 8, 8, c * 2 // 8, 8).transpose(0, 2, 1, 3)
            out_view[:, ::2, :, :] = coef_view[:, :, :, :]
            out_view[:, 1::2, :, :] = coef_view[:, :, :, :]

        # case 4: row scale (X) and col scale (X)
        elif need_scale[i][0] == 2 and need_scale[i][1] == 2:
            out_arr = np.zeros((r, c))
            out_view = out_arr.reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3)
            out_view[:, :, :, :] = coef_view[:, :, :, :]

        else:
            raise KeyError("Something wrong here.")

        DCT_coef[i] = out_arr

    return DCT_coef


# %%
def _calculate_DCT(image_path: str) -> np.ndarray:
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
im_name = "Im_2.jpg"
DCT_coef = read_as_jpeg(IMAGES_PATH + im_name)
im_name = "Im_2.png"
DCT_coef2 = read_as_jpeg(IMAGES_PATH + im_name)

print(DCT_coef.shape, DCT_coef2.shape)

print(DCT_coef[0, :8, :8])
print(DCT_coef2[0, :8, :8])


# %%
images = [cv.imread(IMAGES_PATH + path) for path in os.listdir(IMAGES_PATH)]
image.plot_multiple_images(images=images, titles=os.listdir(IMAGES_PATH), ncols=2)

# %%
image_choice = 1
method_name = "naive"
