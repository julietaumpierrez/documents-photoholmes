# %%
import os

if "research" in os.path.abspath("."):
    os.chdir("../../")
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %%
from photoholmes.methods.focal import Focal, focal_preprocessing

focal = Focal(
    ["ViT", "HRNet"],
    ["weights/focal/ViT_weights.pth", "weights/focal/HRNet_weights.pth"],
)

# %%
from photoholmes.utils.image import read_image

img = read_image("data/test-images/img00.png")
img = focal_preprocessing(image=img)

# %%
out = focal.predict(**img)

# %%
from matplotlib import pyplot as plt

plt.imshow(out)
# %%
