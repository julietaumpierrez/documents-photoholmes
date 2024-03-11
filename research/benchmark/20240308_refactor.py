# %%
import os

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
# %%
from photoholmes.benchmark import Benchmark
from photoholmes.datasets.dataset_factory import DatasetFactory
from photoholmes.methods.dq import DQ, dq_preprocessing
from photoholmes.metrics.metric_factory import MetricFactory

# %%
benchmark = Benchmark(use_existing_output=False, save_extra_outputs=True)

# %%
dataset = DatasetFactory.load(
    dataset_name="columbia",
    dataset_dir="data/Benchmark/Columbia Uncompressed Image Splicing Detection",
    item_data=["image", "image_name", "dct_coefficients", "qtables"],
    transform=dq_preprocessing,
)
# %%
dq = DQ()
# %%
out = benchmark.run(dq, dataset, MetricFactory.load(["auroc", "f1"]))

# %%
out

# %%
from torchmetrics import AUROC, F1Score

benchmark.run(dq, dataset, [AUROC(task="binary"), F1Score(task="binary")])

# %%
