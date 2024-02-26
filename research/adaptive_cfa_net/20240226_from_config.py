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
method = AdaptiveCFANet.from_config("src/photoholmes/methods/adaptive_cfa_net/config.yaml")
# %%
from photoholmes.utils.image import read_image

# %%
image = read_image("data/pelican.png")
image

# %%
image_data = {"image": image, "metadata": {"test": "test"}}
# %%
# %%
input = adaptive_cfa_net_preprocessing(**image_data)
input
# %%
output_1 = method.predict(**input)
output_1
# %%
from photoholmes.utils.image import plot

# %%
plot(output_1["heatmap"])
# %%
