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
from photoholmes.methods.psccnet import PSCCNet, psccnet_preprocessing
from photoholmes.methods.trufor import TruFor, trufor_preprocessing
from photoholmes.methods.zero import Zero, zero_preprocessing

# %%
from photoholmes.utils.image import read_image, read_jpeg_data

# %%
img_path = "/Users/julietaumpierrez/Desktop/Datasets/trace/images/r0a42c0f6t/jpeg_grid_endo.png"
# img_path = "/Users/julietaumpierrez/Desktop/Datasets/Columbia Uncompressed Image Splicing Detection/4cam_splc/nikond70_kodakdcs330_sub_26.tif"
# img_path = "/Users/julietaumpierrez/Desktop/NoiseSniffer/test.png"
# img_path = (
#   "/Users/julietaumpierrez/Desktop/Datasets/tifs-database/DSO-1/splicing-05.png"
# )
dct, qtables = read_jpeg_data(
    img_path,
    num_dct_channels=1,
)
image = read_image(img_path)

# %%
print(qtables)
# %%


image_data = {"image": image, "dct_coefficients": dct, "qtables": qtables}
# %%
input = catnet_preprocessing(**image_data)
# %%
path = "/Users/julietaumpierrez/Desktop/weights/"
method = CatNet(arch_config="pretrained", weights=path + "CAT_full_v2.pth.tar")
method.to_device("mps")

# %%
output_1 = method.predict(**input)
output_1

# %%
import matplotlib.pyplot as plt

plt.imshow(output_1[0])
# %%
output_1[2]
# %%
from photoholmes.methods.psccnet import PSCCNet, psccnet_preprocessing

# Read an image
from photoholmes.utils.image import read_image

path_to_image = (
    "/Users/julietaumpierrez/Desktop/Datasets/tifs-database/DSO-1/splicing-05.png"
)
image = read_image(path_to_image)

# Assign the image to a dictionary and preprocess the image
image_data = {"image": image}
input = psccnet_preprocessing(**image_data)

# Declare the method and use the .to_device if you want to run it on cuda or mps instead of cpu
arch_config = "pretrained"
path = "/Users/julietaumpierrez/Desktop/weights/pscc/"
path_to_weights = {
    "FENet": path + "HRNet.pth",
    "SegNet": path + "NLCDetection.pth",
    "ClsNet": path + "DetectionHead.pth",
}
method = PSCCNet(
    arch_config=arch_config,
    weights_paths=path_to_weights,
)
device = "cpu"
method.to_device(device)

# Use predict to get the final result
output = method.predict(**input)
# %%
import matplotlib.pyplot as plt

plt.imshow(output[0])
# %%
from photoholmes.methods.factory import MethodFactory

method, preprocess = MethodFactory.load(
    "trufor",
    "/Users/julietaumpierrez/Desktop/PhotoHolmesRepo/photoholmes/src/photoholmes/methods/trufor/config.yaml",
)
from photoholmes.utils.image import read_image

img = read_image(
    "/Users/julietaumpierrez/Desktop/Datasets/tifs-database/DSO-1/splicing-05.png"
)
inputs = preprocess(image=img)
out = method.predict(**inputs)

# %%
import matplotlib.pyplot as plt

plt.imshow(out[0])
# %%
