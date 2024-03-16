# %%
import os

from photoholmes.methods.trufor.models.utils import net

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
import cv2
import matplotlib.pyplot as plt
import numpy as np

# %%
# Photoholmes imports
from photoholmes.methods.adaptive_cfa_net import (
    AdaptiveCFANet,
    adaptive_cfa_net_preprocessing,
)
from photoholmes.methods.catnet import CatNet, catnet_preprocessing
from photoholmes.methods.dq import DQ, dq_preprocessing
from photoholmes.methods.exif_as_language import (
    EXIFAsLanguage,
    exif_as_language_preprocessing,
)
from photoholmes.methods.focal import Focal, focal_preprocessing
from photoholmes.methods.noisesniffer import Noisesniffer, noisesniffer_preprocessing
from photoholmes.methods.psccnet import PSCCNet, psccnet_preprocessing
from photoholmes.methods.splicebuster import Splicebuster, splicebuster_preprocessing
from photoholmes.methods.trufor import TruFor, trufor_preprocessing
from photoholmes.methods.zero import Zero, zero_preprocessing
from photoholmes.utils.image import read_image, read_jpeg_data


# %%
# Define overlay function
def overlay(path, output):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Compute the heatmap
    heatmap = output  # * output_1[1].numpy()

    # Normalize the heatmap to 0-255 and convert to 8-bit unsigned integer
    heatmap_normalized = cv2.normalize(
        heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    heatmap_uint8 = np.uint8(heatmap_normalized)

    # Apply the color map
    heatmap_img = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the image
    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)

    # Convert superimposed image from BGR to RGB for plotting
    super_imposed_img_rgb = cv2.cvtColor(super_imposed_img, cv2.COLOR_BGR2RGB)
    return super_imposed_img_rgb


# %%
# Paths
weights_path = "/Users/julietaumpierrez/Desktop/weights/"
image_path = "/Users/julietaumpierrez/Desktop/cvpr_photoholmes/"
# %%
# AdaptiveCFA
image = read_image(image_path + "paul_cvpr.jpeg")
image_data = {"image": image}
input = adaptive_cfa_net_preprocessing(**image_data)
arch_config = "pretrained"
path_to_weights = weights_path + "pretrained.pt"
method = AdaptiveCFANet(
    arch_config=arch_config,
    weights=path_to_weights,
)
output = method.predict(**input)
plt.imshow(output.to("cpu").numpy())
overlay_rgb = overlay(image_path + "paul_cvpr.jpeg", output.to("cpu").numpy())
plt.imshow(overlay_rgb)
plt.imsave(image_path + "paul_cvpr_adaptive_cfa_pretrained.png", overlay_rgb)
# %%
# AdaptiveCFA with jpeg weights
image = read_image(image_path + "paul_cvpr.jpeg")
image_data = {"image": image}
input = adaptive_cfa_net_preprocessing(**image_data)
arch_config = "pretrained"
path_to_weights = weights_path + "adapted_to_j95_database.pt"
method = AdaptiveCFANet(
    arch_config=arch_config,
    weights=path_to_weights,
)
output = method.predict(**input)
plt.imshow(output.to("cpu").numpy())
overlay_rgb = overlay(image_path + "paul_cvpr.jpeg", output.to("cpu").numpy())
plt.imshow(overlay_rgb)
plt.imsave(image_path + "paul_cvpr_adaptive_cfa_jpeg.png", overlay_rgb)
# %%
# CatNet
image = read_image(image_path + "paul_cvpr.jpeg")
dct, qtables = read_jpeg_data(
    image_path + "paul_cvpr.jpeg",
    num_dct_channels=1,
)
image_data = {"image": image, "dct_coefficients": dct, "qtables": qtables}
input = catnet_preprocessing(**image_data)
arch_config = "pretrained"
path_to_weights = weights_path + "CAT_full_v2.pth.tar"
method = CatNet(
    arch_config=arch_config,
    weights=path_to_weights,
)
output = method.predict(**input)
plt.imshow(output[0].to("cpu").numpy())
overlay_rgb = overlay(image_path + "paul_cvpr.jpeg", output[0].to("cpu").numpy())
plt.imshow(overlay_rgb)
plt.imsave(image_path + "paul_cvpr_catnet.png", overlay_rgb)
# %%
# DQ
image = read_image(image_path + "paul_cvpr.jpeg")
dct, qtables = read_jpeg_data(
    image_path + "paul_cvpr.jpeg",
    num_dct_channels=1,
)
image_data = {"image": image, "dct_coefficients": dct, "qtables": qtables}
input = dq_preprocessing(**image_data)
method = DQ()
output = method.predict(**input)
plt.imshow(output)
overlay_rgb = overlay(image_path + "paul_cvpr.jpeg", output)
plt.imshow(overlay_rgb)
plt.imsave(image_path + "paul_cvpr_dq.png", overlay_rgb)
# %%
# Exif
image = read_image(image_path + "paul_cvpr.jpeg")
dct, qtables = read_jpeg_data(
    image_path + "paul_cvpr.jpeg",
    num_dct_channels=1,
)
image_data = {"image": image}
input = exif_as_language_preprocessing(**image_data)
arch_config = "pretrained"
path_to_weights = weights_path + "pruned_weights.pth"
method = EXIFAsLanguage(
    arch_config=arch_config,
    weights=path_to_weights,
)
output = method.predict(**input)
plt.imshow(output[0])
overlay_rgb = overlay(image_path + "paul_cvpr.jpeg", output[0])
plt.imshow(overlay_rgb)
plt.imsave(image_path + "paul_cvpr_exif_ms.png", overlay_rgb)
plt.imshow(output[1])
overlay_rgb = overlay(image_path + "paul_cvpr.jpeg", output[1])
plt.imshow(overlay_rgb)
plt.imsave(image_path + "paul_cvpr_exif_ncuts.png", overlay_rgb)
# %%
# Focal
image = read_image(image_path + "paul_cvpr.jpeg")
image_data = {"image": image}
input = focal_preprocessing(**image_data)
method = Focal(
    weights={
        "ViT": weights_path + "ViT_weights.pth",
        "HRNet": weights_path + "HRNet_weights.pth",
    },
)
output = method.predict(**input)
plt.imshow(output.to("cpu").numpy())
overlay_rgb = overlay(image_path + "paul_cvpr.jpeg", output.to("cpu").numpy())
plt.imshow(overlay_rgb)
plt.imsave(image_path + "paul_cvpr_focal.png", overlay_rgb)
# %%
# Noisesniffer
image = read_image(image_path + "paul_cvpr.jpeg")
image_data = {"image": image}
input = noisesniffer_preprocessing(**image_data)
method = Noisesniffer()
output = method.predict(**input)
plt.imshow(output[0])
overlay_rgb = overlay(image_path + "paul_cvpr.jpeg", output[0])
plt.imshow(overlay_rgb)
plt.imsave(image_path + "paul_cvpr_noisesniffer.png", overlay_rgb)
# %%
# PSCCNet
image = read_image(image_path + "paul_cvpr.jpeg")
image_data = {"image": image}
input = psccnet_preprocessing(**image_data)
arch_config = "pretrained"
path_to_weights = {
    "FENet": weights_path + "pscc/HRNet.pth",
    "SegNet": weights_path + "pscc/NLCDetection.pth",
    "ClsNet": weights_path + "pscc/DetectionHead.pth",
}
method = PSCCNet(
    arch_config=arch_config,
    weights=path_to_weights,
)
output = method.predict(**input)
plt.imshow(output[0].to("cpu").numpy())
overlay_rgb = overlay(image_path + "paul_cvpr.jpeg", output[0].to("cpu").numpy())
plt.imshow(overlay_rgb)
plt.imsave(image_path + "paul_cvpr_psccnet.png", overlay_rgb)
# %%
# Splicebuster gu
image = read_image(image_path + "paul_cvpr.jpeg")
image_data = {"image": image}
input = splicebuster_preprocessing(**image_data)
method = Splicebuster(mixture="uniform")
output = method.predict(**input)
plt.imshow(output)
overlay_rgb = overlay(image_path + "paul_cvpr.jpeg", output)
plt.imshow(overlay_rgb)
plt.imsave(image_path + "paul_cvpr_splicebuster_gu.png", overlay_rgb)


# %%
# Splicebuster gg
# image = read_image(image_path + "paul_cvpr.jpeg")
# image_data = {"image": image}
# input = splicebuster_preprocessing(**image_data)
# method = Splicebuster(mixture="gaussian")
# output = method.predict(**input)
# plt.imshow(output)
# overlay_rgb = overlay(image_path + "paul_cvpr.jpeg", output)
# plt.imshow(overlay_rgb)
# plt.imsave(image_path + "paul_cvpr_splicebuster_gg.png", overlay_rgb)
# %%
# Trufor
image = read_image(image_path + "paul_cvpr.jpeg")
image_data = {"image": image}
input = trufor_preprocessing(**image_data)
arch_config = "pretrained"
path_to_weights = weights_path + "trufor.pth.tar"
method = TruFor(
    arch_config=arch_config,
    weights=path_to_weights,
)
output = method.predict(**input)
plt.imshow(output[0].to("cpu").numpy())
overlay_rgb = overlay(
    image_path + "paul_cvpr.jpeg", (output[0] * output[1]).to("cpu").numpy()
)
plt.imshow(overlay_rgb)
plt.imsave(image_path + "paul_cvpr_trufor.png", overlay_rgb)
# %%
# Zero
image = read_image(image_path + "paul_cvpr.jpeg")
dct, qtables = read_jpeg_data(
    image_path + "paul_cvpr.jpeg",
    num_dct_channels=1,
)
image_data = {"image": image, "dct_coefficients": dct, "qtables": qtables}
input = zero_preprocessing(**image_data)
method = Zero()
output = method.predict(**input)
plt.imshow(output[0])
overlay_rgb = overlay(image_path + "paul_cvpr.jpeg", output[0])
plt.imshow(overlay_rgb)
plt.imsave(image_path + "paul_cvpr_zero.png", overlay_rgb)
# %%
