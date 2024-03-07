# %%
import os

import numpy as np
from matplotlib import pyplot as plt
from requests import get

from photoholmes.methods.splicebuster import Splicebuster, splicebuster_preprocess
from photoholmes.utils.image import read_image

if "research" in os.getcwd():
    os.chdir("../..")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
# %%
image = read_image("data/test-images/img00.png")
input_image = splicebuster_preprocess(image=image)
plt.imshow(input_image["image"])

# %%
sp = Splicebuster()
# %%
out = sp.predict(**input_image)
out["heatmap"].shape

# %%
plt.imshow(out["heatmap"])
# %%
a = np.load("data/orig_mahal_00.npy")
a.shape
# %%
plt.imshow(out["heatmap"])
# %%
plt.imshow(a)

# %%
image = read_image("data/test-images/img03.png")
input_image = splicebuster_preprocess(image=image)
plt.imshow(input_image["image"], cmap="gray")
# %%
out = sp.predict(**input_image)
plt.imshow(out["heatmap"])
# %%
a = np.load("data/orig_mahal_03.npy")
plt.imshow(a)

from skimage.morphology import disk

# %%
from photoholmes.methods.splicebuster.utils import get_disk_kernel

# %%
get_disk_kernel(3) == disk(3)
# %%
