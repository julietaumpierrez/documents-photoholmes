# %%
get_ipython().run_line_magic("load_ext", "autoreload")  # noqa # type: ignore
get_ipython().run_line_magic("autoreload", "2")  # type: ignore # noqa
# %%
# FIXME: No corre acá, solo adentro de src.
import os

import cv2 as cv
import jpegio as jio
import matplotlib.pyplot as plt
import numpy as np

# import matplotlib.pyplot as plt
from photoholmes.models.DQ.method import DQ

DATA_DIR = "/home/pento/workspace/fing/photoholmes/benchmarking/test_images/"
config_yaml = (
    "/home/pento/workspace/fing/photoholmes/src/photoholmes/models/DQ/config.yaml"
)

img_path = (
    "/home/pento/workspace/fing/photoholmes/benchmarking/test_images/images/Im_1.jpg"
)

# %%
jpeg_struct_mod = jio.read(img_path)
coefficients = jpeg_struct_mod.coef_arrays

# %%
coefficients[2].shape

# %%


method = DQ.from_config(config_yaml)

largest_shape = (4608, 3456)
coefficients[0] = np.resize(coefficients[0], largest_shape)
coefficients[1] = np.resize(coefficients[1], largest_shape)
coefficients[2] = np.resize(coefficients[2], largest_shape)

coefficients = np.array(coefficients)
print(coefficients.shape)
heatmap = method.predict(coefficients)
predicted_mask = method.predict_mask(heatmap)

# %%
plt.imshow(heatmap)
# %%
