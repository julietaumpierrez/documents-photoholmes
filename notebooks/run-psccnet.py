from photoholmes.metrics.factory import MetricFactory

metrics = MetricFactory.load(["precision", "f1", "tpr"])

from photoholmes.benchmark import Benchmark

benchmark = Benchmark(
    save_method_outputs=True,
    save_extra_outputs=False,
    save_metrics=True,
    output_folder="supatlantique_benchmark",
    device="cuda",
    use_existing_output=True,
    verbose=1,
)

from photoholmes.datasets.supatlantique import SupatlantiqueDataset
from photoholmes.methods.psccnet import PSCCNet, psccnet_preprocessing

PATH = "/clusteruy/home/julieta.umpierrez/documents-photoholmes/dataset/supatlantique"
arch_config = "pretrained"
path_to_weights = {
    "FENet": "/clusteruy/home/julieta.umpierrez/documents-photoholmes/weights/psccnet/FENet.pth",
    "SegNet": "/clusteruy/home/julieta.umpierrez/documents-photoholmes/weights/psccnet/SegNet.pth",
    "ClsNet": "/clusteruy/home/julieta.umpierrez/documents-photoholmes/weights/psccnet/ClsNet.pth",
}

psccnet = PSCCNet(
    arch_config=arch_config,
    weights=path_to_weights,
)
dataset = SupatlantiqueDataset(
    dataset_path=PATH,
    load=["image", "dct_coefficients"],
    preprocessing_pipeline=psccnet_preprocessing,
)
psccnet_results = benchmark.run(
    method=psccnet,
    dataset=dataset,
    metrics=metrics,
)
print(psccnet_results)
