# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image  # type: ignore
from skimage import io

from photoholmes.models.noisesniffer.method import Noisesniffer


# %%
def read_image(filename):
    """
    Reads an image with three-channels.
    Input: filename
    Ourput: a three-channels image
    """
    img_original = io.imread(filename).astype(np.float32)
    if len(img_original.shape) == 2 or img_original.shape[2] == 1:
        img = np.zeros((img_original.shape[0], img_original.shape[1], 3))
        img[:, :, 0] = img_original
        img[:, :, 1] = img_original
        img[:, :, 2] = img_original
    else:
        img = img_original[:, :, :3]
    return img


# %%
image = Image.open(
    "/Users/julietaumpierrez/Desktop/NoiseSniffer/test.png"
)  # .convert("L")
np_image = np.array(image)  # .astype(float)
plt.imshow(np_image, cmap="gray")
# filename = "/Users/julietaumpierrez/Desktop/NoiseSniffer/test.png"
# np_image = read_image(filename)
# %%
print(np.unique(np_image))
# %%

noisesniffer = Noisesniffer()

# %%
mask, mapita = noisesniffer.predict(np_image)
# %%
plt.imshow(mask)
# %%
image_orig = Image.open("/Users/julietaumpierrez/Desktop/NoiseSniffer/output_mask.png")
# %%
plt.imshow(image_orig - mask)
# %%
# %%
image = Image.open(
    "/Users/julietaumpierrez/Desktop/NoiseSniffer/test.png"
)  # .convert("L")
np_image = np.array(image).astype(float)
image = Image.open(
    "/Users/julietaumpierrez/Desktop/NoiseSniffer/test.png"
)  # .convert("L")
np_image64 = np.array(image).astype(np.float32)

plt.imshow(np_image64 - np_image)
print(np.unique(np_image64))
# print()


# %%
