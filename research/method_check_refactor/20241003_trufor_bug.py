# %%
import os
from turtle import title

import matplotlib.pyplot as plt

from photoholmes.methods.trufor.models.utils import net

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

from photoholmes.methods.psccnet import PSCCNet, psccnet_preprocessing

# %%
from photoholmes.methods.trufor import TruFor, trufor_preprocessing
from photoholmes.utils.image import read_image, read_jpeg_data

# %%
img_path = (
    "/Users/julietaumpierrez/Desktop/Datasets/tifs-database/DSO-1/splicing-25.png"
)
image = read_image(img_path)
image_data = {"image": image}
input = trufor_preprocessing(**image_data)

# %%
# CPU
path = "/Users/julietaumpierrez/Desktop/weights/trufor.pth.tar"
method = TruFor(weights=path)
method.to_device("cpu")
output_1 = method.predict(**input)
plt.imshow(output_1[0])
plt.title("CPU")
plt.show()

# %%
image = read_image(img_path)
image_data = {"image": image.to("mps")}
input = trufor_preprocessing(**image_data)
path = "/Users/julietaumpierrez/Desktop/weights/trufor.pth.tar"
method = TruFor(weights=path)
method.to_device("mps")
output_1 = method.predict(**input)
plt.imshow(output_1[0].cpu().numpy())
plt.title("MPS")
plt.show()
# %%
img_path = (
    "/Users/julietaumpierrez/Desktop/Datasets/trace/images/r0a42c0f6t/jpeg_grid_exo.png"
)
# img_path = "/Users/julietaumpierrez/Desktop/Datasets/Columbia Uncompressed Image Splicing Detection/4cam_splc/nikond70_kodakdcs330_sub_26.tif"
image = read_image(img_path)
image_data = {"image": image}
input = trufor_preprocessing(**image_data)
path = "/Users/julietaumpierrez/Desktop/weights/pscc/"
arch_config = "pretrained"
path_to_weights = {
    "FENet": path + "HRNet.pth",
    "SegNet": path + "NLCDetection.pth",
    "ClsNet": path + "DetectionHead.pth",
}
method = PSCCNet(
    arch_config=arch_config,
    weights_paths=path_to_weights,
)
method.to_device("cpu")
output_1_cpu = method.predict(**input)
plt.imshow(output_1_cpu[0])
plt.title("CPU")
# %%
# img_path = "/Users/julietaumpierrez/Desktop/Datasets/trace/images/r0a42c0f6t/cfa_grid_endo.png"
image = read_image(img_path)
image_data = {"image": image}
input = trufor_preprocessing(**image_data)
path = "/Users/julietaumpierrez/Desktop/weights/pscc/"
arch_config = "pretrained"
path_to_weights = {
    "FENet": path + "HRNet.pth",
    "SegNet": path + "NLCDetection.pth",
    "ClsNet": path + "DetectionHead.pth",
}
method = PSCCNet(
    arch_config=arch_config,
    weights_paths=path_to_weights,
)
method.to_device("mps")
output_1_mps = method.predict(**input)
plt.imshow(output_1_mps[0].to("cpu").numpy())
plt.title("MPS")

# %%
diff = output_1_mps[0].to("cpu") - output_1_cpu[0]
# check if outputs are close enough
assert diff.abs().max() < 1e-4
# %%
plt.imshow(diff)
# %%
import numpy as np

np.unique(diff)
# %%
