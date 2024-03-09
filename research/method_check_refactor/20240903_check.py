# %%
import os

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
# %%
from photoholmes.methods.adaptive_cfa_net import (
    AdaptiveCFANet,
    adaptive_cfa_net_preprocessing,
)

# %%
from photoholmes.utils.image import read_image

# %%
image = read_image(
    "/Users/julietaumpierrez/Desktop/Datasets/trace/images/r0a42c0f6t/cfa_grid_endo.png"
)

# %%
image_data = {"image": image}
# %%
input = adaptive_cfa_net_preprocessing(**image_data)
# %%
method = AdaptiveCFANet(
    arch_config="pretrained",
    weights="/Users/julietaumpierrez/Desktop/weights/pretrained.pt",
)
method.to_device("mps")

# %%
output_1 = method.predict(**input)
output_1

# %%
import matplotlib.pyplot as plt

plt.imshow(output_1.to("cpu").numpy())
# %%
