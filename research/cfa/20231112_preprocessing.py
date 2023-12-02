# %%
import os

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
# %%
import torch

from src.photoholmes.models.cfa.method import CFANet
from src.photoholmes.models.cfa.preprocessing import CFANetPreprocessing
from src.photoholmes.utils.image import plot, read_image

# %%
model = CFANet.from_config(
    {"weights": "src/photoholmes/models/cfa/weights/pretrained.pt"}
)
model.eval()
# %%
img_path = "benchmarking/test_images/images/canonxt_kodakdcs330_sub_29.tif"
img = read_image(img_path)
img.shape
# %%
img = CFANetPreprocessing()(img)

# %%
img["x"].shape
# %%
with torch.no_grad():
    out = model.predict(**img)

# %%
out.shape
# %%
plot(out)

# %%
img["x"].shape
# %%
