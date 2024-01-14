# %%
import os

if "research" in os.getcwd():
    os.chdir("..")
    os.chdir("..")
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

# %%
from photoholmes.methods.method_factory import MethodFactory

trufor, preprocess = MethodFactory.load("trufor", "weights/trufor/config.yaml")
trufor.eval()
# %%
from photoholmes.utils.image import read_image

img = read_image("data/img00.png")
# %%
p_image = preprocess(image=img)
# %%
out = trufor.predict(**p_image)
# %%
import matplotlib.pyplot as plt

plt.imshow(out["heatmap"].squeeze())

# %%
