# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from photoholmes.models.exif_as_language.method import EXIFAsLanguage, preprocess

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
# %%
image = Image.open(
    # "/Users/julietaumpierrez/Downloads/Columbia Uncompressed Image Splicing Detection/4cam_auth/canonxt_35_sub_03.tif"
    # "data/COLUMBIA/4cam_auth/canonxt_35_sub_03.tif",
    "data/img00.png",
)  # .convert("L")
# image = Image.open("data/img00.png")  # .convert("L")
np_image = np.array(image)
plt.imshow(np_image, cmap="gray")
# %%
device = "cpu"
# device = "mps"
path = "weights/exif/pruned_weights.pth"
# path = "/Users/julietaumpierrez/Desktop/Exif_as_language/exif-as-language/pruned_weights.pth"
model = EXIFAsLanguage(
    transformer="distilbert", visual="resnet50", state_dict_path=path, device=device
)
torched_image = torch.from_numpy(np_image).permute(2, 0, 1)
preprocessed_image = preprocess(torched_image)
# %%
pred = model.predict(preprocessed_image)
# %%
plt.imshow(pred["ms"])
# %%
plt.imshow(pred["ncuts"])
# %%
