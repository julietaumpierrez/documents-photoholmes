# %%
import os

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
# %%
from src.photoholmes.data.input.dataset_factory import DatasetFactory
from src.photoholmes.metrics.metric_factory import MetricFactory
from src.photoholmes.models.method_factory import MethodFactory

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
from src.photoholmes.utils.preprocessing.image import ToTensor

# %%
columbia = DatasetFactory.load(
    "columbia",
    "data/Columbia_subsample/",
    tampered_only=True,
    transform=dq_preprocessing,
    item_data=["image", "dct_coefficients", "qtables"],
)
# %%
columbia[0]
# %%
os.listdir("data/Columbia_subsample")
# %%
