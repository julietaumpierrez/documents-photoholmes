# %%
import os
from typing import cast

import yaml

from photoholmes.models.catnet.config import (
    CatnetArchConfig,
    StageConfig,
    pretrained_arch,
)

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %%
config = yaml.load(open("weights/catnet.yaml", "r"), Loader=yaml.FullLoader)["MODEL"][
    "EXTRA"
]
catnet_config = cast(CatnetArchConfig, config)

# %%
pretrained_arch.__dict__["STAGE1"].__dict__["NUM_BLOCKS"] = [-1]
# %%
conf = CatnetArchConfig.load_from_dict(config)


# %%
def test(**kwargs):
    for k, v in kwargs.items():
        print(k, v)


# %%
test(**pretrained_arch.__dict__)
# %%
import numpy as np

from photoholmes.models.catnet import CatNet, catnet_preprocessing
from photoholmes.utils.image import read_jpeg_data

model = CatNet.from_config(
    {"arch": "pretrained", "weights": "weights/CAT_full_v2.pth.tar"}
)


# %%
import matplotlib.pyplot as plt
from PIL import Image

image_path = "data/example_input.jpg"
img = np.array(Image.open(image_path))
dct, qtable_ph = read_jpeg_data(image_path, num_channels=1)
t_x_ph, t_qtable_ph = catnet_preprocessing(img, dct, qtable_ph, n_dct_channels=1)
mask = model.predict(t_x_ph[None, :], t_qtable_ph[None, :])

mask = model.predict(t_x_ph[None, :], t_qtable_ph[None, :])
plt.imshow(mask[0])
# %%
