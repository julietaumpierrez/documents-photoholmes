# $$
import os

if "research" in os.path.abspath("."):
    os.chdir("../../")

import warnings

# %%
from photoholmes.benchmark.model import Benchmark
from photoholmes.datasets.dataset_factory import DatasetFactory
from photoholmes.methods.method_factory import MethodFactory
from photoholmes.metrics.metric_factory import MetricFactory

warnings.filterwarnings("ignore")

method_name = "splicebuster"
config = f"src/photoholmes/methods/{method_name}/config.yaml"
device = "cpu"

method, preprocessing = MethodFactory.load(
    method_name=method_name, config=config, device=device
)
base_path = "/home/pento/workspace/fing/datasets/"
dataset_dict = {
    # "dso1": f"{base_path}DSO-1/tifs-database",
    # "columbia": f"{base_path}Columbia",
    # "casia1_copy_move": f"{base_path}CASIA_1",
    # "casia1_splicing": f"{base_path}CASIA_1",
    "realistic_tampering": f"{base_path}realistic-tampering",
    "autosplice_100": f"{base_path}AutoSplice",
    "autosplice_90": f"{base_path}AutoSplice",
    "autosplice_75": f"{base_path}AutoSplice",
    "trace_noise_exo": f"{base_path}minitrace",
    "trace_noise_endo": f"{base_path}minitrace",
    "trace_cfa_alg_exo": f"{base_path}minitrace",
    "trace_cfa_alg_endo": f"{base_path}minitrace",
    "trace_cfa_grid_exo": f"{base_path}minitrace",
    "trace_cfa_grid_endo": f"{base_path}minitrace",
    "trace_jpeg_grid_exo": f"{base_path}minitrace",
    "trace_jpeg_grid_endo": f"{base_path}minitrace",
    "trace_jpeg_quality_exo": f"{base_path}minitrace",
    "trace_jpeg_quality_endo": f"{base_path}minitrace",
    "trace_hybrid_exo": f"{base_path}minitrace",
    "trace_hybrid_endo": f"{base_path}minitrace",
    "coverage": f"{base_path}COVERAGE",
}


for dataset_name, dataset_path in dataset_dict.items():
    if dataset_name in ["columbia", "casia1_copy_move", "casia1_splicing"]:
        tampered = [False, True]
    elif dataset_name in [
        "coverage",
        "realistic_tampering",
        "autosplice_100",
        "autosplice_90",
        "autosplice_75",
        "dso1",
    ]:
        tampered = [False]
    else:
        tampered = [True]
    for tampered_only in tampered:
        dataset = DatasetFactory.load(
            dataset_name=dataset_name,
            dataset_dir=dataset_path,
            tampered_only=tampered_only,
            transform=preprocessing,
            item_data=["image_name", "image", "original_image_size"],
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
            device=device,
            verbose=1,
            save_metrics=False,
        )

        benchmark.run(
            method=method,
            dataset=dataset,
            metrics=metrics,
        )
