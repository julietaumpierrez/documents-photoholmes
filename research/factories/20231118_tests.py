# %%
import os

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
# %%
from src.photoholmes.datasets.dataset_factory import DatasetFactory
from src.photoholmes.methods.method_factory import MethodFactory
from src.photoholmes.metrics.metric_factory import MetricFactory

# %%
# load the model
dq, dq_preprocessing = MethodFactory.load("dq")
# %%
dq
# %%
dq_preprocessing.transforms
# %%
columbia = DatasetFactory.load(
    "columbia",
    "/home/pento/workspace/fing/photoholmes/data/Columbia_subsample",
    tampered_only=True,
    transform=dq_preprocessing,
    item_data=["image", "dct_coefficients", "qtables"],
)
# %%
# make sure the dataset is loaded correctly
columbia[0]

# %%
coverage = DatasetFactory.load(
    "coverage",
    "data/COVERAGE",
    tampered_only=True,
    transform=dq_preprocessing,
    item_data=["dct_coefficients"],
)
# %%
coverage[0]
# %%
osn = DatasetFactory.load(
    "osn",
    "data/osn",
    tampered_only=False,
    transform=dq_preprocessing,
    item_data=["dct_coefficients"],
)
# %%
osn[0]
# %%
realistic_tampering = DatasetFactory.load(
    "realistic_tampering",
    "data/realistic-tampering-dataset",
    tampered_only=True,
    transform=dq_preprocessing,
    item_data=["dct_coefficients"],
)
# %%
realistic_tampering[0]
# %%
import torch

# test the model on the dataset
preditcion1 = torch.tensor(dq.predict(**columbia[0][0]))[:, :757]
# %%
iou = MetricFactory.load("iou")
# %%
mask1 = columbia[0][1]
# %%
# make the mask and the prediction the same shape

preditcion1_t = preditcion1 > 0.2
# %%
from src.photoholmes.utils.image import plot

# %%
plot(mask1)
plot(preditcion1)
# %%
iou(preditcion1_t, mask1)
# %%
auroc = MetricFactory.load("auroc")
# %%
auroc(preditcion1, mask1)

# %%
