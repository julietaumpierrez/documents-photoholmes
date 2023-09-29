# %%
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from photoholmes.models.splicebuster.method import Splicebuster

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_like_magic("load_ext autoreload")
    get_ipython().run_like_magic("autoreload 2")
# %%
image = Image.open("data/img00.png").convert("L")
np_image = np.array(image)

# %%
sp = Splicebuster(stride=8)

# %%
heatmap = sp.predict(np_image)
# %%
plt.imshow(heatmap[0])
plt.figure()
plt.imshow(heatmap[1])

# %%
