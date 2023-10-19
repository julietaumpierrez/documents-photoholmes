# %%
import os

import numpy as np
from PIL import Image

from photoholmes.models.splicebuster import Splicebuster

if "research" in os.path.abspath("."):
    os.chdir("../../")
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %%
image = Image.open("data/img00.png").convert("L")
np_image = np.array(image) / 255

# %%
model = Splicebuster(mixture="uniform")

# %%
heatmap = model.predict(np_image)
comp_heatmap = 1 / heatmap
comp_heatmap /= comp_heatmap.max()

# %%
import matplotlib.pyplot as plt

plt.imshow(heatmap)

# %%
heatmap.max(), (comp_heatmap).max()
# %%
heatmap.sum(), (comp_heatmap).sum()
