# %%
import os

from matplotlib import pyplot as plt

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

img = read_image("data/test-images/img00.png")

# %%
inputs = preprocess(image=img)
out = trufor.predict(**inputs)

# %%
print("heatmap", out["heatmap"].shape)
print("confidence", out["confidence"].shape)
print("detection", out["detection"].shape, out["detection"])
# %%
print("npp", out["noiseprints"].shape, out["noiseprints"])

# %%
plt.imshow(out["heatmap"])
# %%
noiseprint = out["noiseprints"][0].permute(1, 2, 0)
noiseprint = (noiseprint - noiseprint.min()) / (noiseprint.max() - noiseprint.min())
plt.imshow(noiseprint)

# %%
import numpy as np

res = np.load(
    "/home/joflaherty/Documents/Proyecto/photoholmes/output/trufor/coveragedataset/outputs/1/arrays.npz"
)
# %%
noiseprint = res["noiseprint"][0].transpose(1, 2, 0)
plt.imshow(noiseprint)
# %%
