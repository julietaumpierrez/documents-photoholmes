# %%
import os

import torch
import yaml
from cv2 import threshold

from photoholmes.models.catnet import CatNet, catnet_preprocessing
from photoholmes.models.catnet.config import pretrained_config

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
# %%
model = CatNet("pretrained", threshold=0.1)
model.load_weigths("weights/CAT_full_v2.pth.tar")
# %%
weights = torch.load("weights/CAT_full_v2.pth.tar", map_location="cpu")
model = CatNet(pretrained_config, 2)
model.load_weigths(weights)
# %%
config = yaml.load(open("weights/catnet.yaml", "r"), Loader=yaml.FullLoader)["MODEL"][
    "EXTRA"
]
model = CatNet(config)
# %%
