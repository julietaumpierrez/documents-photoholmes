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
method_name = "noisesniffer"
config = "src/photoholmes/methods/noisesniffer/config.yaml"
device = "cpu"

method, preprocessing = MethodFactory.load(
    method_name=method_name, config=config, device=device
)

# %%
dataset_name = "columbia"
dataset_path = "/Users/julietaumpierrez/Desktop/Datasets/Columbia Uncompressed Image Splicing Detection/"
tampered_only = True
dataset = DatasetFactory.load(
    dataset_name=dataset_name,
    dataset_dir=dataset_path,
    tampered_only=tampered_only,
    transform=preprocessing,
)

# %%
metrics = MetricFactory.load(
    [
        "auroc",
        "fpr",
        "iou",
        "mcc",
        "precision",
        "roc",
        "tpr",
        "f1_weighted",
        "iou_weighted",
        "mcc_weighted",
    ]
)

# %%
from photoholmes.benchmark.model import Benchmark

# %%
benchmark = Benchmark(
    save_output=True,
    output_path="output/",
    device="cpu",
)
# %%
benchmark.run(
    method=method,
    dataset=dataset,
    metrics=metrics,
)
# %%
