import json
import logging
import os
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from photoholmes.datasets.dataset_factory import DatasetFactory
from photoholmes.datasets.registry import DatasetName
from photoholmes.methods.method_factory import MethodFactory
from photoholmes.methods.registry import MethodName
from photoholmes.metrics.metric_factory import MetricFactory
from photoholmes.metrics.registry import MetricName

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger("benchmark.model.Benchmark")
log.setLevel(logging.INFO)


class Benchmark:
    # Add documentation to class and methods
    def __init__(
        self,
        method_name: MethodName,
        method_config: Optional[Union[dict, str]],
        dataset_name: DatasetName,
        dataset_path: str,
        tampered_only: bool,
        metrics_names: list[MetricName],
        save_output: bool = False,
        output_path: str = "output/",
        device: str = "cpu",
    ):
        # TODO: add a method to send the method to the device
        self.method_config = method_config
        self.method, self.preprocessing = MethodFactory.load(
            method_name=method_name, config=method_config, device=device
        )

        self.dataset = DatasetFactory.load(
            dataset_name=dataset_name,
            dataset_dir=dataset_path,
            tampered_only=tampered_only,
            transform=self.preprocessing,
        )
        self.mask_metrics = MetricFactory.load(metrics_names)
        self.heatmap_metrics = MetricFactory.load(metrics_names)
        self.save_output = save_output
        self.output_path = os.path.join(
            output_path, method_name.value, dataset_name.value
        )

        self.mask_metrics_names = metrics_names
        self.method_name = method_name
        self.dataset_name = dataset_name
        if device.startswith("cuda") and not torch.cuda.is_available():
            log.warning(
                f"Requested device '{device}' is not available. Falling back to 'cpu'."
            )
            device = "cpu"
        self.device = torch.device(device)
        log.info(f"Using device: {self.device}")

        # TODO: set an attribute "output_keys" in the method class and use that
        # to determine whether to save the mask and heatmap or not
        self.save_mask = False
        self.save_heatmap = False

    def run(self):
        log.info("-" * 80)
        log.info("Running the benchmark")
        log.info("Benchmark configuration:")
        log.info(f"    Method: {self.method_name.value}")
        log.info(f"    Method config: {self.method_config}")

        log.info(f"    Dataset: {self.dataset_name.value}")
        log.info(f"    Metrics: {[metric.value for metric in self.mask_metrics_names]}")
        log.info(f"    Output path: {self.output_path}")
        log.info(f"    Save output: {self.save_output}")
        log.info(f"    Device: {self.device}")
        log.info("-" * 80)

        for data, mask, image_name in tqdm(self.dataset, desc="Processing Images"):
            # TODO: make a cleaner way to move the data to the device
            # (conditioned to the method or something)
            data_on_device = self.move_to_device(data)

            output = self.method.predict(**data_on_device)

            self.update_metrics(output, mask)

            if self.save_output:
                self.save_pred_output(image_name, output)

        log.info("-" * 80)
        log.info("Saving metrics")
        self.save_metrics()
        log.info("Metrics saved")
        log.info("-" * 80)
        log.info("Benchmark finished")
        log.info("-" * 80)
        log.info("-" * 80)

    def move_to_device(self, data):
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in data.items()
        }

    def update_metrics(self, output, mask):
        if isinstance(mask, torch.Tensor):
            mask = mask.to(self.device)

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
            log.info("     Saving mask metrics")
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
            log.info("     No mask metrics to save")

        if self.save_heatmap:
            log.info("     Saving heatmap metrics")
            heatmap_metrics_report = self.heatmap_metrics.compute()
            torch.save(
                self.heatmap_metrics.state_dict(),
                os.path.join(metrics_path, "heatmap_state.pt"),
            )
            heatmap_metrics_report = {
                key: float(value) for key, value in heatmap_metrics_report.items()
            }
            with open(os.path.join(metrics_path, "heatmap_report.json"), "w") as f:
                json.dump(heatmap_metrics_report, f)
        else:
            log.info("     No heatmap metrics to save")

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
