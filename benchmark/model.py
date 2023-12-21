import json
import os

import numpy as np
import torch

from photoholmes.datasets.dataset_factory import DatasetFactory
from photoholmes.methods.method_factory import MethodFactory
from photoholmes.metrics.metric_factory import MetricFactory
from photoholmes.metrics.registry import MetricName


class Benchmark:
    def __init__(
        self,
        method_name: str,
        dataset_name: str,
        dataset_path: str,
        tampered_only: bool,
        metrics_names: list[str | MetricName],
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

        self.mask_metrics_names = metrics_names
        self.method_name = method_name
        self.dataset_name = dataset_name

        # TODO: set an attribute "output_keys" in the method class and use that
        # to determine whether to save the mask or not
        self.save_mask = False
        self.save_heatmap = False

    def run(self):
        print("-" * 80)
        print("-" * 80)
        print("Running the benchmark")
        print("Benchmark configuration:")
        print(f"    Method: {self.method_name}")
        print(f"    Dataset: {self.dataset_name}")
        print(f"    Metrics: {self.mask_metrics_names}")
        print(f"    Output path: {self.output_path}")
        print(f"    Save output: {self.save_output}")

        print("-" * 80)

        for data, mask, image_name in self.dataset:
            output = self.method.predict(**data)

            self.update_metrics(output, mask)

            if self.save_output:
                self.save_pred_output(image_name, output)

        print("Saving metrics")
        self.save_metrics()

        print("Benchmark finished")
        print("-" * 80)
        print("-" * 80)

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
