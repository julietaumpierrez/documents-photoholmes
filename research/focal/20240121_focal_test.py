# %%
import os

if "research" in os.path.abspath("."):
    os.chdir("../../")
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %%
from photoholmes.methods.focal import Focal, focal_preprocessing

focal = Focal.from_config("src/photoholmes/methods/focal/config.yaml", "cuda")

# %%
from photoholmes.utils.image import read_image

img = read_image(
    "/home/pento/workspace/fing/datasets/minitrace/r0b3220e1t/noise_endo.png"
)
img = focal_preprocessing(image=img)

# %%
out = focal.predict(**img)

# %%
from matplotlib import pyplot as plt

image = out["heatmap"]
plt.imshow(image)
# %%
