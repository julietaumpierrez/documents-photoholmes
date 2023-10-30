# %%
import os

import cv2 as cv
import matplotlib.pyplot as plt

from photoholmes.models.method_factory import MethodFactory
from photoholmes.utils import image

os.chdir("..")

DATA_DIR = "benchmarking/test_images/"

IMAGES_PATH = DATA_DIR + "images/"
MASK_PATH = DATA_DIR + "masks/"

# %%
images = [cv.imread(IMAGES_PATH + path) for path in os.listdir(IMAGES_PATH)]
image.plot_multiple(images=images, titles=os.listdir(IMAGES_PATH), ncols=2)

# %%
image_choice = 1
method_name = "naive"

method = MethodFactory.create(method_name)
name = f"Im_{image_choice}"
im = cv.imread(IMAGES_PATH + name + ".jpg")
mask = cv.imread(MASK_PATH + name + ".png")

image.plot(im, name)

# %%

heatmap = method.predict(im)
predicted_mask = method.predict_mask(heatmap)

# %%
imgs_to_plot = [im, heatmap, predicted_mask, mask]
titles = ["Imagen", "Heatmap", "Predicted Mask", "GT Mask"]
image.plot_multiple(imgs_to_plot, titles)

# %%
