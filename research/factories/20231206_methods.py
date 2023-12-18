# %%
import os

import numpy as np

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
from src.photoholmes.datasets.dataset_factory import DatasetFactory
from src.photoholmes.methods.method_factory import MethodFactory
from src.photoholmes.utils.image import plot, plot_multiple

# %%
model, preprocessing = MethodFactory.load(
    "splicebuster",
)
# %%
dataset = DatasetFactory.load(
    "columbia",
    "data/Columbia_subsample",
    transform=preprocessing,
)

# %%
img_data, mask = dataset[0]
img_data.keys()
# %%
# %%
prediction = model.predict(**img_data)
prediction.keys()
# %%
if "heatmap" in prediction:
    assert mask.shape == prediction["heatmap"].shape
# %%
if "mask" in prediction:
    print(prediction["mask"].shape)
    print(mask.shape)
    assert mask.shape == prediction["mask"].shape
# %%
heatmap = np.array(prediction["heatmap"])
# pred_mask = np.array(prediction["mask"])
mask = np.array(mask)
plot_multiple([heatmap, mask])
# %%
prediction["score"]
# %%
