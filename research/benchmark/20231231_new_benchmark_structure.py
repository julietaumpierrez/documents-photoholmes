# %%
import os

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
# %%
from photoholmes.datasets.dataset_factory import DatasetFactory
from photoholmes.methods.method_factory import MethodFactory
from photoholmes.metrics.metric_factory import MetricFactory

# %%
# load the model
method_name = "catnet"
config = "src/photoholmes/methods/catnet/config.yaml"
device = "cpu"

method, preprocessing = MethodFactory.load(
    method_name=method_name, config=config, device=device
)

# %%
dataset_name = "columbia"
dataset_path = "data/Columbia"
tampered_only = True
dataset = DatasetFactory.load(
    dataset_name=dataset_name,
    dataset_dir=dataset_path,
    tampered_only=tampered_only,
    transform=preprocessing,
)

# %%
metrics = MetricFactory.load(["precision", "FPR"])

# %%
from photoholmes.benchmark.model import Benchmark
# %%
benchmark = Benchmark(
    save_output=True,
    output_path="output/",
    device="cuda:0",
)
# %%
benchmark.run(
    method=method,
    dataset=dataset,
    metrics=metrics,
)
# %%
