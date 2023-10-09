# %%
import os

import cv2 as cv
import jpeglib
import matplotlib.pyplot as plt
import numpy as np
from PIL.Image import open

from photoholmes.models.method_factory import MethodFactory
from photoholmes.utils import image

os.chdir("..")

DATA_DIR = "benchmarking/test_images/"

IMAGES_PATH = DATA_DIR + "images/"
MASK_PATH = DATA_DIR + "masks/"

# %%
im_name = "Im_2.png"
os.listdir()
im_read = open(IMAGES_PATH + im_name)
im_read.save(
    IMAGES_PATH + im_name[:-4] + ".jpg",
)
image.plot_multiple_images([im_read], im_name)

# %%
im_name = "Im_2.jpg"
dct = jpeglib.read_dct(IMAGES_PATH + im_name)
print(dct.qt)

# %%
images = [cv.imread(IMAGES_PATH + path) for path in os.listdir(IMAGES_PATH)]
image.plot_multiple_images(images=images, titles=os.listdir(IMAGES_PATH), ncols=2)

# %%
image_choice = 1
method_name = "naive"
