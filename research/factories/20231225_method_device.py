# %%
import os

from torchmetrics import Metric

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %%
from photoholmes.methods.method_factory import MethodFactory

# %%
method, preprocessing = MethodFactory.load("dq", device="cpu")
# %%
method.model_to_device()

# %%
method.device
# %%
method.name

method.device
# %%
from photoholmes.datasets.dataset_factory import DatasetFactory

# %%
dataset = DatasetFactory.load("columbia", "data/columbia")
# %%
dataset.class
# %%
from photoholmes.metrics.metric_factory import MetricFactory

# %%
metric = MetricFactory.load(["precision","FPR"])
# %%
metric._get_name()
# %%
for met in metric:
    print(met)
# %%
