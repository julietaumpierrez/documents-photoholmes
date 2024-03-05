# %%
import os

import torch
from matplotlib import pyplot as plt

if "research" in os.getcwd():
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
# %%
from photoholmes.methods.psccnet import PSCCNet, psccnet_preprocessing
from photoholmes.utils.image import read_image

img = read_image("data/test-images/img00.png")
# %%
psccnet = PSCCNet(
    arch_config="pretrained",
    weights_paths={
        "FENet": "weights/pscnet/HRNet.pth",
        "SegNet": "weights/pscnet/NLCDetection.pth",
        "ClsNet": "weights/pscnet/DetectionHead.pth",
    },
    device="cpu",
)

# %%
input_image = psccnet_preprocessing(image=img)

# %%
heatmap, detection = psccnet.predict(**input_image)
# out = psccnet.predict(**input_image)
# heatmap = out["heatmap"]
# detection = out["detection"]
# %%
plt.imshow(heatmap)
plt.show()
print("Detection Score:", detection)
# %%
benchmark_output = psccnet.benchmark(**input_image)

# %%
plt.imshow(benchmark_output["heatmap"])
plt.show()
print("Detection Score:", benchmark_output["detection"])

# %%
old_mask = torch.load("data/old-mask-psccnet.pth")
# %%
assert (old_mask == heatmap).all()
