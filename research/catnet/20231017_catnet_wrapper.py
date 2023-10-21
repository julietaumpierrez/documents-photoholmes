# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

from photoholmes.models.catnet import CatNet, catnet_preprocessing
from photoholmes.models.catnet.config import pretrained_arch
from photoholmes.utils.image import read_jpeg_data

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
# %%
model = CatNet("pretrained", threshold=0.1)
model.load_weigths("weights/CAT_full_v2.pth.tar")
# %%
weights = torch.load("weights/CAT_full_v2.pth.tar", map_location="cpu")
model = CatNet(pretrained_arch, 2)
model.load_weigths(weights)
# %%
config = yaml.load(open("weights/catnet.yaml", "r"), Loader=yaml.FullLoader)["MODEL"][
    "EXTRA"
]
model = CatNet(config)
weights = torch.load("weights/CAT_full_v2.pth.tar", map_location="cpu")
model.load_state_dict(weights["state_dict"])
# %%
model = CatNet(pretrained_arch, 2, weights="weights/CAT_full_v2.pth.tar")

# %%
image_path = "data/example_input.jpg"

img = np.array(Image.open(image_path))
dct, qtable_ph = read_jpeg_data(image_path, num_channels=1)
t_x_ph, t_qtable_ph = catnet_preprocessing(img, dct, qtable_ph, n_dct_channels=1)
plt.imshow(t_x_ph[:3].permute(1, 2, 0).numpy())

# %%
model.train()
mask = model.predict(t_x_ph[None, :], t_qtable_ph[None, :])
plt.imshow(mask[0])
# %%
model.eval()
mask = model.predict(t_x_ph[None, :], t_qtable_ph[None, :])
plt.imshow(mask[0])
