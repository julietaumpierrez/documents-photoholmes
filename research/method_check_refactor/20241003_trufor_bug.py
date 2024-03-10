# %%
import os
from turtle import title

import matplotlib.pyplot as plt

from photoholmes.methods.trufor.models.utils import net

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %%
from photoholmes.methods.trufor import TruFor, trufor_preprocessing
from photoholmes.utils.image import read_image, read_jpeg_data

# %%
img_path = (
    "/Users/julietaumpierrez/Desktop/Datasets/tifs-database/DSO-1/splicing-05.png"
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
