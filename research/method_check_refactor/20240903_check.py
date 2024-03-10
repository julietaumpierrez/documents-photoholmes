# %%
import os

from photoholmes.methods.trufor.models.utils import net

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
# %%
from photoholmes.methods.catnet import CatNet, catnet_preprocessing
from photoholmes.methods.exif_as_language import (
    EXIFAsLanguage,
    exif_as_language_preprocessing,
)
from photoholmes.methods.focal import Focal, focal_preprocessing
from photoholmes.methods.noisesniffer import Noisesniffer, noisesniffer_preprocessing

# %%
from photoholmes.utils.image import read_image, read_jpeg_data

# %%
# img_path = "/Users/julietaumpierrez/Desktop/Datasets/trace/images/r0a42c0f6t/noise_endo.png"
# img_path = "/Users/julietaumpierrez/Desktop/Datasets/Columbia Uncompressed Image Splicing Detection/4cam_splc/nikond70_kodakdcs330_sub_26.tif"
img_path = "/Users/julietaumpierrez/Desktop/NoiseSniffer/test.png"
dct, qtables = read_jpeg_data(
    img_path,
    num_dct_channels=1,
)
image = read_image(img_path)

# %%
print(qtables)
# %%


image_data = {"image": image}  # , "dct_coefficients": dct, "qtables": qtables}
# %%
input = noisesniffer_preprocessing(**image_data)
# %%
method = NoiseSniffer()
method.to_device("cpu")

# %%
output_1 = method.predict(**input)
output_1

# %%
import matplotlib.pyplot as plt

plt.imshow(output_1)
# %%
