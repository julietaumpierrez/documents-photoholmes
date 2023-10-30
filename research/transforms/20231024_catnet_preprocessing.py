# %%
import os

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from photoholmes.models.method_factory import MethodFactory
from photoholmes.utils.image import read_jpeg_data

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %%
method, preprocessing = MethodFactory.create(
    "catnet", config={"weights": "weights/CAT_full_v2.pth.tar"}
)
# %%
image = np.array(Image.open("data/img00.png"))
dct, qtables = read_jpeg_data("data/img00.png", num_dct_channels=1)
plt.imshow(image)

# %%
if preprocessing:
    input_image = preprocessing(image=image, dct_coefficients=dct, qtables=qtables)


# %%
mask = method.predict(**input_image)
# %%
plt.imshow(mask[0])
# %%
