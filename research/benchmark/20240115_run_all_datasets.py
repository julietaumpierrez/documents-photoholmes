import os

if "research" in os.path.abspath("."):
    os.chdir("../../")
from photoholmes.benchmark.model import Benchmark
from photoholmes.datasets.dataset_factory import DatasetFactory
from photoholmes.methods.method_factory import MethodFactory
from photoholmes.metrics.metric_factory import MetricFactory

method_name = "exif_as_language"
config = f"src/photoholmes/methods/{method_name}/config.yaml"
device = "mps"

method, preprocessing = MethodFactory.load(
    method_name=method_name, config=config, device=device
)
base_path = "/Users/sote/Desktop/data/datasets/"
dataset_dict = {
    # "dso1": f"{base_path}DSO-1/tifs-database",
    "columbia": f"{base_path}COLUMBIA", # Se rompe
    "coverage": f"{base_path}COVERAGE",
    "casia1_copy_move": f"{base_path}CASIA_1",
    "casia1_splicing": f"{base_path}CASIA_1",
    "realistic_tampering": f"{base_path}realistic_tampering",
    
    "autosplice_100": f"{base_path}AutoSplice",
    "autosplice_90": f"{base_path}AutoSplice",
    "autosplice_75": f"{base_path}AutoSplice",
    
    # "trace_noise_exo": f"{base_path}minitrace",
    # "trace_noise_endo": f"{base_path}minitrace",
    "trace_cfa_alg_exo": f"{base_path}minitrace", # Se rompe
    "trace_cfa_alg_endo": f"{base_path}minitrace",
    "trace_cfa_grid_exo": f"{base_path}minitrace",
    "trace_cfa_grid_endo": f"{base_path}minitrace",
    "trace_jpeg_grid_exo": f"{base_path}minitrace",
    "trace_jpeg_grid_endo": f"{base_path}minitrace",
    "trace_jpeg_quality_exo": f"{base_path}minitrace",
    "trace_jpeg_quality_endo": f"{base_path}minitrace",
    "trace_hybrid_exo": f"{base_path}minitrace",
    "trace_hybrid_endo": f"{base_path}minitrace",
    
    
}

for dataset_name, dataset_path in dataset_dict.items():
    tampered_only = True
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
        device=device,
    )

    benchmark.run(
        method=method,
        dataset=dataset,
        metrics=metrics,
    )
