import os

if "research" in os.path.abspath("."):
    os.chdir("../../")

import warnings

from photoholmes.benchmark import Benchmark
from photoholmes.datasets.factory import DatasetFactory
from photoholmes.methods.factory import MethodFactory
from photoholmes.metrics.factory import MetricFactory

warnings.filterwarnings("ignore")

method_name = "zero"
config = f"src/photoholmes/methods/{method_name}/config.yaml"
device = "cpu"

method, preprocessing = MethodFactory.load(method_name=method_name, config=config)
method.to_device(device)
base_path = "/home/dsense/extra/tesis/datos/"
dataset_dict = {
    # "dso1": f"{base_path}DSO-1-20240214T131337Z-001/DSO-1/tifs-database",
    # "dso1_osn": f"{base_path}DSO_Facebook-20240214T131343Z-001",
    # "columbia": f"{base_path}columbia",
    # "columbia_osn": f"{base_path}columbia_osn",
    "columbia_webp": f"{base_path}columbia_webp",
    # "casia1_copy_move": f"{base_path}casia1/CASIA 1.0 dataset photoholmes",
    # "casia1_splicing": f"{base_path}casia1/CASIA 1.0 dataset photoholmes",
    # "realistic_tampering": f"{base_path}realistic_tampering",
    "realistic_tampering_webp": f"{base_path}realistic_tampering_webp",
    # "autosplice_100": f"{base_path}AutoSplice",
    # "autosplice_90": f"{base_path}AutoSplice",
    # "autosplice_75": f"{base_path}AutoSplice",
    # "trace_noise_exo": f"{base_path}miniTrace/images",
    # "trace_noise_endo": f"{base_path}miniTrace/images",
    # "trace_cfa_alg_exo": f"{base_path}miniTrace/images",
    # "trace_cfa_alg_endo": f"{base_path}miniTrace/images",
    # "trace_cfa_grid_exo": f"{base_path}miniTrace/images",
    # "trace_cfa_grid_endo": f"{base_path}miniTrace/images",
    # "trace_jpeg_grid_exo": f"{base_path}miniTrace/images",
    # "trace_jpeg_grid_endo": f"{base_path}miniTrace/images",
    # "trace_jpeg_quality_exo": f"{base_path}miniTrace/images",
    # "trace_jpeg_quality_endo": f"{base_path}miniTrace/images",
    # "trace_hybrid_exo": f"{base_path}miniTrace/images",
    # "trace_hybrid_endo": f"{base_path}miniTrace/images",
    # "coverage": f"{base_path}coverage",
}


for dataset_name, dataset_path in dataset_dict.items():
    if dataset_name in [
        "dso1",
        "columbia",
        "columbia_webp",
        "casia1_copy_move",
        "casia1_splicing",
    ]:
        tampered = [False, True]
    elif dataset_name in [
        "coverage",
        "realistic_tampering",
        "realistic_tampering_webp",
        "dso1",
    ]:
        tampered = [False]
    else:
        tampered = [True]
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
            save_method_outputs=True,
            output_folder="output/",
            device=device,
            verbose=1,
            save_metrics=True,
        )

        benchmark.run(
            method=method,
            dataset=dataset,
            metrics=metrics,
        )
