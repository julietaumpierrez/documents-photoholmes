# %%
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL.Image import open

from photoholmes.models.method_factory import MethodFactory
from photoholmes.utils import image

os.chdir("..")

DATA_DIR = "benchmarking/test_images/"

IMAGES_PATH = DATA_DIR + "images/"
# %%
method_name = "dq"

im_selection = -1


def is_jpeg(path: str) -> bool:
    extension = (path[-4:]).lower()
    return (extension == ".jpg") or (extension == ".jpeg")


im_names = ["Im_1.jpg", "Im_3.jpg"]
method = MethodFactory.create(method_name)


im_path = IMAGES_PATH + im_names[im_selection]
img = np.asarray(open(im_path)).copy()
image.plot_multiple([img], ncols=2)

# %%
dct_coefs, _ = image.read_jpeg_data(im_path, num_channels=1)
heatmap = method.predict(dct_coefs)
mask_pred = method.predict_mask(heatmap)

image.plot_multiple([img, heatmap, mask_pred], ["Imagen", "BPPM", "Máscara"], ncols=3)

# %%
