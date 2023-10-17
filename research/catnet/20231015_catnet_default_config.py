# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from photoholmes.models.catnet.config import pretrain_config
from photoholmes.models.catnet.model import CAT_Net
from photoholmes.models.catnet.preprocessing import catnet_preprocessing
from photoholmes.utils.image import read_jpeg_data

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %%
import yaml

config = yaml.load(open("weights/catnet.yaml", "r"), Loader=yaml.FullLoader)
config = config["MODEL"]["EXTRA"]

# %%
model = CAT_Net(pretrain_config, 2)
# %%
weights = torch.load("weights/CAT_full_v2.pth.tar", map_location="cpu")
model.load_state_dict(weights["state_dict"])
model.eval()
# %%
image_path = "data/example_input.jpg"

img = np.array(Image.open(image_path))
dct, qtable_ph = read_jpeg_data(image_path, num_channels=1)
t_x_ph, t_qtable_ph = catnet_preprocessing(img, dct, qtable_ph, n_dct_channels=1)
plt.imshow(t_x_ph[:3].permute(1, 2, 0).numpy())
# %%
from extra.catnet.dataset import TestDataset

dataset = TestDataset(
    crop_size=None, grid_crop=True, blocks=["RGB", "DCTvol", "qtable"], DCT_channels=1
)
x, _, qtable = dataset._create_tensor(image_path, None)
# %%
assert (x == t_x_ph).all()
assert (qtable == t_qtable_ph).all()
# %%
with torch.no_grad():
    out = model(t_x_ph[None, :], t_qtable_ph[None, :])
    pred = out.squeeze(0)
    labels = torch.nn.functional.softmax(pred, dim=0)[1]

# %%
plt.imshow(labels.numpy())

# %%
