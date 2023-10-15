# %%
import os

import matplotlib.pyplot as plt
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
sp = Splicebuster()
# %%
mask = sp.predict(np_image)
# %%
plt.imshow(mask)

# %%
plt.imshow((mask > 0.5) * np_image)
