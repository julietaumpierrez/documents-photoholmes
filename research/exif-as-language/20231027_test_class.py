# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from photoholmes.models.exif_as_language.method import EXIF_SC
from PIL import Image

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
# %%
image = Image.open(
    "data/COLUMBIA/4cam_splc/canong3_canonxt_sub_01.tif"
)  # .convert("L")
# image = Image.open("data/img00.png")  # .convert("L")
np_image = np.array(image)
plt.imshow(np_image, cmap="gray")
# %%
device = "cpu"
# device = "mps"
path = "weights/exif/pruned_weights.pth"
# path = "/Users/julietaumpierrez/Desktop/Exif_as_language/exif-as-language/pruned_weights.pth"
model = EXIF_SC(
    transformer="distilbert", visual="resnet50", state_dict_path=path, device=device
)
torched_image = torch.from_numpy(np_image).permute(2, 0, 1)
# %%
pred = model.predict(torched_image)
# %%
plt.imshow(pred["ms"])
# %%
plt.imshow(pred["ncuts"])
# %%
