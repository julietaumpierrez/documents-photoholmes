# %%
import os

if "research" in os.path.abspath("."):
    os.chdir("../../")

import warnings

# %%
from photoholmes.benchmark import Benchmark
from photoholmes.datasets.factory import DatasetFactory
from photoholmes.methods.factory import MethodFactory
from photoholmes.metrics.factory import MetricFactory

warnings.filterwarnings("ignore")

method_name = "splicebuster"
config = f"src/photoholmes/methods/{method_name}/config.yaml"
device = "cpu"

method, preprocessing = MethodFactory.load(method_name=method_name, config=config)
base_path = "/Users/julietaumpierrez/Desktop/Datasets/"
dataset_dict = {
    # "dso1": f"{base_path}DSO-1/tifs-database",
    # "casia1_copy_move": f"{base_path}CASIA_1",
    # "columbia": f"{base_path}Columbia",
    # "trace_noise_exo": f"{base_path}minitrace",
    # "trace_noise_endo": f"{base_path}minitrace",
    # "trace_cfa_alg_exo": f"{base_path}minitrace",
    # "trace_cfa_alg_endo": f"{base_path}minitrace",
    # "trace_cfa_grid_exo": f"{base_path}minitrace",
    # "trace_cfa_grid_endo": f"{base_path}minitrace",
    # "trace_jpeg_grid_exo": f"{base_path}minitrace",
    # "trace_jpeg_grid_endo": f"{base_path}minitrace",
    # "trace_jpeg_quality_exo": f"{base_path}minitrace",
    # "trace_jpeg_quality_endo": f"{base_path}minitrace",
    "trace_hybrid_exo": f"{base_path}trace/images/",
    "trace_hybrid_endo": f"{base_path}/trace/images",
    # "casia1_splicing": f"{base_path}CASIA_1",
    # "realistic_tampering": f"{base_path}realistic-tampering",
    # "autosplice_100": f"{base_path}AutoSplice",
    # "autosplice_90": f"{base_path}AutoSplice",
    # "autosplice_75": f"{base_path}AutoSplice",
    # "coverage": f"{base_path}COVERAGE",
    # "columbia_osn": f"{base_path}osn",
    # "dso1_osn": f"{base_path}osn",
    # "casia1_copy_move_osn": f"{base_path}osn",
    # "casia1_splicing_osn": f"{base_path}osn",
    # "realistic_tampering_webp": f"{base_path}realistic_tampering_webp",
}

for dataset_name, dataset_path in dataset_dict.items():
    if dataset_name in [
        "trace_noise_exo",
        "trace_noise_endo",
        "trace_cfa_alg_exo",
        "trace_cfa_alg_endo",
        "trace_cfa_grid_exo",
        "trace_cfa_grid_endo",
        "trace_jpeg_grid_exo",
        "trace_jpeg_grid_endo",
        "trace_jpeg_quality_exo",
        "trace_jpeg_quality_endo",
        "trace_hybrid_exo",
        "trace_hybrid_endo",
        "columbia_osn",
        "dso1_osn",
        "casia1_copy_move_osn",
        "casia1_splicing_osn",
        "autosplice_100",
        "autosplice_90",
        "autosplice_75",
    ]:
        tampered = [True]
    else:
        tampered = [True, False]
    for tampered_only in tampered:
        dataset = DatasetFactory.load(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            tampered_only=tampered_only,
            preprocessing_pipeline=preprocessing,
            load=["image"],
        )

        metrics = MetricFactory.load(
            [
                "auroc",
                "mauroc",
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
            device=device,
            verbose=1,
            save_metrics=True,
        )

        benchmark.run(
            method=method,
            dataset=dataset,
            metrics=metrics,
        )

# import numpy as np

# # %%
# from photoholmes.utils.image import plot

# %%
# Load the results
# output =np.load("../../output/splicebuster/autosplice100dataset/outputs/39436/output.npz", allow_pickle=True)
# output
# # %%
# plot(output["heatmap"])
# # %%
# output["heatmap"].shape
# %%
dataset = DatasetFactory.load(
    dataset_name=dataset_name,
    dataset_path=dataset_path,
    tampered_only=tampered_only,
    preprocessing_pipeline=preprocessing,
    load=["image"],
)
img, mask, name = dataset[465]
# %%
name
# %%
