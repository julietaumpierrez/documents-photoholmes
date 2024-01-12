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
method_name = "dq"

method, preprocessing = MethodFactory.load(method_name=method_name)

# %%
dataset_name = "columbia"
dataset_path = "data/Columbia"
tampered_only = False
dataset = DatasetFactory.load(
    dataset_name=dataset_name,
    dataset_dir=dataset_path,
    tampered_only=tampered_only,
    transform=preprocessing,
)

# %%


# %%
import json

import numpy as np
import torch


# %%
class Benchmark:
    def __init__(
        self,
        method_name,
        dataset_name,
        dataset_path,
        tampered_only,
        metrics_names,
        save_output: bool = False,
        output_path: str = "output/",
    ):
        self.method, self.preprocessing = MethodFactory.load(method_name=method_name)
        self.dataset = DatasetFactory.load(
            dataset_name=dataset_name,
            dataset_dir=dataset_path,
            tampered_only=tampered_only,
            transform=self.preprocessing,
        )
        self.mask_metrics = MetricFactory.load(metrics_names)
        self.heatmap_metrics = MetricFactory.load(metrics_names)
        self.save_output = save_output
        self.output_path = os.path.join(output_path, method_name, dataset_name)

        # TODO: set an attribute "output_keys" in the method class and use that
        # to determine whether to save the mask or not
        self.save_mask = False
        self.save_heatmap = False

    def run(self):
        for data, mask, image_name in self.dataset:
            output = self.method.predict(**data)

            self.update_metrics(output, mask)

            if self.save_output:
                self.save_pred_output(image_name, output)

        self.save_metrics()

    def update_metrics(self, output, mask):
        if "mask" in output:
            self.mask_metrics.update(output["mask"], mask)

            # TODO: delete next line when the "output_keys" attribute is set
            # in the method class and use that to determine whether to save the mask
            # or not
            self.save_mask = True
        if "heatmap" in output:
            self.heatmap_metrics.update(output["heatmap"], mask)

            # TODO: delete next line when the "output_keys" attribute is set
            # in the method class and use that to determine whether to save the heatmap
            # or not
            self.save_heatmap = True

    def save_metrics(self):
        metrics_path = os.path.join(self.output_path, "metrics")
        os.makedirs(metrics_path, exist_ok=True)

        if self.save_mask:
            print("Saving mask metrics")
            mask_metrics_report = self.mask_metrics.compute()
            torch.save(
                self.mask_metrics.state_dict(),
                os.path.join(metrics_path, "mask_state.pt"),
            )
            mask_metrics_report = {
                key: float(value) for key, value in mask_metrics_report.items()
            }
            with open(os.path.join(metrics_path, "mask_report.json"), "w") as f:
                json.dump(mask_metrics_report, f)
        else:
            print("No mask metrics to save")

        if self.save_heatmap:
            print("Saving heatmap metrics")
            heatmap_metrics_report = self.heatmap_metrics.compute()
            torch.save(
                self.heatmap_metrics.state_dict(),
                os.path.join(metrics_path, "heatmap_state.pt"),
            )
            with open(os.path.join(metrics_path, "heatmap_report.json"), "w") as f:
                heatmap_metrics_report = {
                    key: float(value) for key, value in heatmap_metrics_report.items()
                }
                json.dump(heatmap_metrics_report, f)
        else:
            print("No heatmap metrics to save")

    def save_pred_output(self, image_name, output):
        image_save_path = os.path.join(self.output_path, "outputs", image_name)
        os.makedirs(image_save_path, exist_ok=True)

        array_like_dict = {}
        non_array_like_dict = {}

        for key, value in output.items():
            if isinstance(value, (torch.Tensor, list, np.ndarray)):
                array_like_dict[key] = value
            else:
                non_array_like_dict[key] = value

        np.savez_compressed(os.path.join(image_save_path, "arrays"), **array_like_dict)
        if non_array_like_dict:
            with open(os.path.join(image_save_path, "data.json"), "w") as f:
                json.dump(non_array_like_dict, f)


# %%
benchmark = Benchmark(
    method_name="dq",
    dataset_name="columbia",
    dataset_path="data/Columbia",
    tampered_only=True,
    metrics_names=["FPR", "AUROC"],
    save_output=True,
    output_path="output/",
)
# %%
benchmark.run()
# %%
report = benchmark.heatmap_metrics.compute()
report
# %%
# turn the values of the report into floats instead of being torch tensors
report = {key: float(value) for key, value in report.items()}
report
# %%
bench
