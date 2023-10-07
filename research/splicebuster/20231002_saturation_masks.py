# %%
import os
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

if "research" in os.path.abspath("."):
    os.chdir("../../")
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")


# %%
img = np.array(Image.open("data/img00.png")) / 255
plt.imshow(img)


# %%
def get_saturaded_pixels_basic(
    img: np.ndarray, low_th: float = 6 / 255, high_th: float = 252 / 255
):
    img = img.transpose(2, 0, 1)
    mask = np.array([(ch < low_th) + (ch > high_th) for ch in img])
    mask = reduce(np.logical_or, mask)
    return ~mask


mask = get_saturaded_pixels_basic(img)
plt.imshow(mask * 255)
# %%
plt.imshow(mask[:, :, None] * img)

# %%
import scipy as sp


def get_disk_kernel(radius: int):
    xx, yy = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    kernel = (xx**2 + yy**2) <= radius**2
    return 1.0 * kernel


# %%
radius = 3
mesh = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
kernel = (mesh[0] ** 2 + mesh[1] ** 2) <= radius**2


# %%

from scipy.ndimage import binary_dilation, binary_opening


def get_saturated_pixels_processed(
    img: np.ndarray,
    low_th: float = 6 / 255,
    high_th: float = 252 / 255,
    kernel_radius: int = 3,
):
    if img.ndim == 2:
        img = img[:, :, None]
    img = img.transpose(2, 0, 1)
    kernel = get_disk_kernel(kernel_radius)
    mask_low = np.array(
        [binary_opening(ch < low_th, kernel) for ch in img], dtype=np.float32
    )
    mask_high = np.array(
        [binary_opening(ch > high_th, kernel) for ch in img], dtype=np.float32
    )
    mask = mask_low + mask_high
    mask = reduce(np.logical_or, mask)
    mask = binary_dilation(mask, np.ones((9, 9)))
    return ~mask


mask = get_saturated_pixels_processed(img)
plt.imshow(mask, cmap="gray")

# %%
plt.imshow(mask[:, :, None] * img)

# %%
img = Image.open("data/img00.png").convert("L")
img = np.array(img) / 255
plt.imshow(img, cmap="gray")

mask = get_saturated_pixels_processed(img)
plt.figure()
plt.imshow(mask, cmap="gray")
# %%
