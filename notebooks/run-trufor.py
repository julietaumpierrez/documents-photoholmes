from photoholmes.metrics.factory import MetricFactory

metrics = MetricFactory.load(["precision", "f1", "tpr"])

from photoholmes.benchmark import Benchmark

benchmark = Benchmark(
    save_method_outputs=True,
    save_extra_outputs=False,
    save_metrics=True,
    output_folder="supatlantique_benchmark",
    device="cpu",
    use_existing_output=True,
    verbose=1,
)

from photoholmes.datasets.supatlantique import SupatlantiqueDataset
from photoholmes.methods.trufor import TruFor, trufor_preprocessing

PATH = "/clusteruy/home/julieta.umpierrez/documents-photoholmes/dataset/supatlantique"
weights = "/clusteruy/home/julieta.umpierrez/documents-photoholmes/weights/trufor/trufor.pth.tar"
# PATH = "/Users/julietaumpierrez/Desktop/iccv-doc-workshop/supatlantique"
# weights = "/Users/julietaumpierrez/Desktop/PhotoHolmesRepo/photoholmes/weights/trufor/trufor.pth.tar"
trufor = TruFor(weights=weights)
dataset = SupatlantiqueDataset(
    dataset_path=PATH,
    load=["image", "dct_coefficients"],
    preprocessing_pipeline=trufor_preprocessing,
)
trufor_results = benchmark.run(
    method=trufor,
    dataset=dataset,
    metrics=metrics,
)
print(trufor_results)
