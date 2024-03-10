# %%
import os

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
# %%
from photoholmes.methods.catnet import CatNet, catnet_preprocessing

# %%
from photoholmes.utils.image import read_image, read_jpeg_data

# %%
# img_path = "/Users/julietaumpierrez/Desktop/Datasets/trace/images/r0a42c0f6t/jpeg_quality_endo.png"
img_path = "/Users/julietaumpierrez/Desktop/Datasets/Columbia Uncompressed Image Splicing Detection/4cam_splc/nikond70_kodakdcs330_sub_26.tif"
dct, qtables = read_jpeg_data(
    img_path,
    num_dct_channels=1,
)
image = read_image(img_path)

# %%
print(qtables)
# %%
import torch

image_data = {"image": image, "dct_coefficients": dct, "qtables": qtables}
# %%
input = catnet_preprocessing(**image_data)
# %%
method = CatNet(
    arch_config="pretrained",
    weights="/Users/julietaumpierrez/Desktop/weights/CAT_full_v2.pth.tar",
)
method.to_device("mps")

# %%
output_1 = method.predict(**input)
output_1

# %%
import matplotlib.pyplot as plt

plt.imshow(output_1[0].to("cpu").numpy())
# %%
