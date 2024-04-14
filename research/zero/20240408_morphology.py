# %%
import os

if "research" in os.getcwd():
    os.chdir("../..")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %%
import matplotlib.pyplot as plt

from photoholmes.methods.zero import Zero, zero_preprocessing
from photoholmes.utils.image import read_image

# img = read_image("data/COVERAGE/image/26t.tif")
img = read_image("../implementations/zero/tampered1.png")
img = zero_preprocessing(image=img)["image"]
plt.imshow(img)

# %%
zero = Zero()
pred, _, _ = zero.predict(img)
plt.imshow(pred)
# %%
import time

import numpy as np


def custom_closing(image, W=9):
    Y, X = image.shape

    image_out = np.zeros_like(image)
    mask_aux = np.zeros_like(image)
    for x in range(W, X - W):
        for y in range(W, Y - W):
            if image[y, x] != 0:
                mask_aux[y - W : y + W + 1, x - W : x + W + 1] = 1
                image_out[y - W : y + W + 1, x - W : x + W + 1] = 1

    for x in range(W, X - W):
        for y in range(W, Y - W):
            if mask_aux[y, x] == 0:
                image_out[y - W : y + W + 1, x - W : x + W + 1] = 0

    return image_out


t0 = time.time()
pred_closed = custom_closing(pred)
tf = time.time()
print("Elapsed time:", tf - t0)
plt.imshow(pred_closed)

# %%
from scipy.ndimage import binary_closing

t0 = time.time()
pred_close_scipy = binary_closing(pred, structure=np.ones((9, 9)))
t1 = time.time()
print("Elapsed time:", t1 - t0)
plt.imshow(pred_close_scipy)

# %%
assert (pred_closed == pred_close_scipy).all()

# %%
plt.imshow(pred_closed != pred_close_scipy)

# %%
import cv2

t0 = time.time()
pred_close_cv2 = cv2.morphologyEx(
    pred.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9))
)
t1 = time.time()
print("Elapsed time:", t1 - t0)
plt.imshow(pred_close_cv2)

assert (pred_close_scipy == pred_close_cv2).all()
assert (pred_closed == pred_close_cv2).all()


# %%
def custom_closing_2(image, W=9):
    Y, X = image.shape
    image_out = np.zeros_like(image)
    mask_aux = np.zeros_like(image)
    mask_aux[W:-W, W:-W] = image[W:-W, W:-W] != 0
    image_out[W:-W, W:-W] = mask_aux[W:-W, W:-W]
    image_out = np.where(mask_aux == 0, 0, image_out)
    return image_out


t0 = time.time()
pred_closed = custom_closing_2(pred)
tf = time.time()
print("Elapsed time:", tf - t0)
plt.imshow(pred_closed)

# %%
np.where()
