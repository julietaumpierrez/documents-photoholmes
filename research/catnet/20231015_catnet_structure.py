# %%
import os
from typing import Optional

import torch
import torch.nn as nn

from photoholmes.models.catnet.model import CAT_Net

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
# %%
weights = torch.load("weights/CAT_full_v2.pth.tar", map_location="cpu")

# %%
import yaml

config = yaml.load(open("weights/catnet.yaml", "r"), Loader=yaml.FullLoader)
# %%
from dataclasses import dataclass

# %
model = CAT_Net(config["MODEL"]["EXTRA"], num_classes=config["DATASET"]["NUM_CLASSES"])

# %%
model.load_state_dict(weights["state_dict"])
model.eval()
# %%
model(torch.randn(1, 24, 512, 512), torch.randn(1, 8, 8))

# %%
