# %%
import os

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
# %%

import numpy as np
import torch

from src.photoholmes.models.cfa.method import FullNet

# %%
model = FullNet.from_config("src/photoholmes/models/cfa/config.yaml")
model.eval()
# %%

# %%
from src.photoholmes.utils.image import plot, read_image

# %%
img_path = "benchmarking/test_images/images/canonxt_kodakdcs330_sub_29.tif"
img = read_image(img_path)
img.shape
# %%

plot(img)
# %%

# %%
C, Y_o, X_o = img.shape
img = img[:3, : Y_o - Y_o % 2, : X_o - X_o % 2]
img.shape
# %%
# turn img to torch.float32

img = img.float()

# %%

# %%
block_size = 32
with torch.no_grad():
    out = model.predict(img.unsqueeze(0))
# %%
out.shape
# %%
res = out.numpy()
res.shape
# %%
res[:, 1] = res[([1, 0, 3, 2], 1)]
res[:, 2] = res[([2, 3, 0, 1], 2)]
res[:, 3] = res[([3, 2, 1, 0], 3)]
res.shape
# %%
res = np.mean(res, axis=1)
res.shape
# %%
best_grid = np.argmax(np.mean(res, axis=(1, 2)))
authentic = np.argmax(res, axis=(0)) == best_grid
confidence = 1 - np.max(res, axis=0)
confidence[confidence < 0] = 0
confidence[confidence > 1] = 1
confidence[authentic] = 1
confidence.shape
# %%
confidence = confidence.repeat(block_size, axis=0).repeat(
    block_size, axis=1
)  # Make it the same size as image
plot(confidence)
# %%
