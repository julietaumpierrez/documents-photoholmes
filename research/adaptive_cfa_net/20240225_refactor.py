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
image = read_image("data/tampered1.png")
image

# %%
image_data = {"image": image, "metadata": {"test": "test"}}
# %%
adaptive_cfa_net_preprocessing(**image_data)
# %%
method = AdaptiveCFANet(
    arch_config="pretrained", weights="weights/adaptive_cfa_net/pretrained.pt"
)
method
# %%
input = adaptive_cfa_net_preprocessing(**image_data)
input
# %%
output_1 = method.predict(**input)
output_1
# %%
output_1["heatmap"].max()
# %%
from photoholmes.utils.image import plot

# %%
plot(output_1["heatmap"])
# %%
from photoholmes.methods.adaptive_cfa import AdaptiveCFANet

# %%
method = AdaptiveCFANet(weights="weights/adaptive_cfa_net/pretrained.pt")
method
# %%
input = adaptive_cfa_net_preprocessing(**image_data)
input
# %%
# change the name of the key image_size to original_image_size
input["original_image_size"] = input.pop("image_size")
input
# %%
output = method.predict(**input)
# %%
plot(output["heatmap"])
# %%
# compare the two outputs
import torch

torch.allclose(output_1["heatmap"], output["heatmap"])  # True
# %%
