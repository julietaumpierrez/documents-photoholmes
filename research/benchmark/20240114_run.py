import os

if "research" in os.path.abspath("."):
    os.chdir("../../")
from photoholmes.benchmark.model import Benchmark
from photoholmes.datasets.dataset_factory import DatasetFactory
from photoholmes.methods.method_factory import MethodFactory
from photoholmes.metrics.metric_factory import MetricFactory

method_name = "splicebuster"
config = f"src/photoholmes/methods/{method_name}/config.yaml"
device = "cpu"

method, preprocessing = MethodFactory.load(
    method_name=method_name, config=config, device=device
)

dataset_name = "columbia"
dataset_path = "data/Columbia"
tampered_only = False
dataset = DatasetFactory.load(
    dataset_name=dataset_name,
    dataset_dir=dataset_path,
    tampered_only=tampered_only,
    transform=preprocessing,
)

metrics = MetricFactory.load(
    [
        "auroc",
        "fpr",
        "iou",
        "mcc",
        "precision",
        "roc",
        "tpr",
        "f1_weighted_v1",
        "iou_weighted_v1",
        "mcc_weighted_v1",
        "f1_weighted_v2",
        "iou_weighted_v2",
        "mcc_weighted_v2",
        "f1",
    ]
)

benchmark = Benchmark(
    save_output=True,
    output_path="output/",
    device="cpu",
)

benchmark.run(
    method=method,
    dataset=dataset,
    metrics=metrics,
)
