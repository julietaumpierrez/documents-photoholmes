# %%
import os

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %%
from matplotlib import pyplot as plt

from photoholmes.methods.noisesniffer import Noisesniffer, noisesniffer_preprocess
from photoholmes.utils.image import read_image

img = read_image("data/test-images/img00.png")
# %%
ns = Noisesniffer()
# %%
input_image = noisesniffer_preprocess(image=img)
mask, _ = ns.predict(**input_image)
# %%
plt.imshow(mask)
# %%
