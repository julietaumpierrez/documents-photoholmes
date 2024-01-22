import json
import logging
import os
import time
from typing import Literal

import numpy as np
import torch
from torchmetrics import MetricCollection
from tqdm import tqdm

from photoholmes.datasets.base import BaseDataset
from photoholmes.methods.base import BaseMethod

logging.basicConfig(format="%(levelname)s - %(message)s")
IO_MESSAGE = 11
logging.addLevelName(IO_MESSAGE, "IO_MESSAGE")


def io_message(self, message, *args, **kws):
    if self.isEnabledFor(IO_MESSAGE):
        self._log(IO_MESSAGE, message, args, **kws)


logging.Logger.io_message = io_message
log = logging.getLogger(__name__)

verbose_dict = {
    0: logging.WARNING,
    1: logging.INFO,
    2: IO_MESSAGE,
}


class Benchmark:
    # Add documentation to class and methods
    def __init__(
        self,
        save_output: bool = True,
        save_metrics: bool = True,
        output_path: str = "output/",
        device: str = "cpu",
        use_existing_output: bool = True,
        verbose: Literal[0, 1, 2] = 1,
    ):
        self.save_output_flag = save_output
        self.save_metrics_flag = save_metrics
        self.output_path = output_path
        self.use_existing_output = use_existing_output
        self.verbose = verbose
        if self.verbose not in verbose_dict:
            log.warning(
                f"Invalid verbose level '{self.verbose}'. "
                f"Using default verbose level '1'."
            )
            self.verbose = 1
        log.setLevel(verbose_dict[self.verbose])

        if device.startswith("cuda") and not torch.cuda.is_available():
            log.warning(
                f"Requested device '{device}' is not available. Falling back to 'cpu'."
            )
        self.device = torch.device("cpu")
        log.info(f"Using device: {self.device}")

        # TODO: set an attribute "output_keys" in the method class and use that
        # to determine whether to save the mask and heatmap or not
        self.save_mask = False
        self.save_heatmap = False
        self.save_detection = False

    def run(self, method: BaseMethod, dataset: BaseDataset, metrics: MetricCollection):
        if method.device != self.device:
            log.warning(
                f"Method device '{method.device}' does not match benchmark device '{self.device}'. "
                f"Moving method to '{self.device}'"
            )
            method.method_to_device(self.device)

        output_path = os.path.join(
            self.output_path,
            method.__class__.__name__.lower(),
            dataset.__class__.__name__.lower(),
        )

        log.info("-" * 80)
        log.info("Running the benchmark")
        log.info("Benchmark configuration:")
        log.info(f"    Method: {method.__class__.__name__}")
        log.info(f"    Dataset: {dataset.__class__.__name__}")
        log.info("    Metrics:")
        for metric in metrics:
            log.info(f"       - {metric}")
        log.info(f"    Output path: {output_path}")
        log.info(f"    Save output flag: {self.save_output_flag}")
        log.info(f"    Save metrics flag: {self.save_metrics_flag}")
        log.info(f"    Device: {self.device}")
        log.info(f"    Check existing output: {self.use_existing_output}")
        log.info(f"    Verbose: {logging._levelToName[verbose_dict[self.verbose]]}")
        log.info("-" * 80)
        if self.save_metrics_flag:
            metrics_on_device = metrics.to("cpu", dtype=torch.float32)
            heatmap_metrics = metrics_on_device.clone(prefix="heatmap")
            mask_metrics = metrics_on_device.clone(prefix="mask")
            detection_metrics = metrics_on_device.clone(prefix="detection")
        for data, mask, image_name in tqdm(dataset, desc="Processing Images"):
            # TODO: make a cleaner way to move the data to the device
            # (conditioned to the method or something)
            output = None
            if self.use_existing_output:
                output = self.check_for_existing_output(output_path, image_name)

            if output is None:
                data_on_device = self.move_to_device(data)
                output = method.predict(**data_on_device)

            output = {
                k: v.to("cpu") if isinstance(v, torch.Tensor) else v
                for k, v in output.items()
            }
            if self.save_metrics_flag:
                mask = mask.to("cpu")
                if "detection" in output:
                    detection_gt = (
                        torch.tensor(int(torch.any(mask))).unsqueeze(0).to("cpu")
                    )
                    detection_metrics.update(output["detection"], detection_gt)
                    self.save_detection = True
                if "mask" in output:
                    mask_metrics.update(output["mask"], mask)
                    self.save_mask = True
                if "heatmap" in output:
                    heatmap_metrics.update(output["heatmap"], mask)
                    self.save_heatmap = True

            if self.save_output_flag:
                self.save_pred_output(output_path, image_name, output)

        log.info("-" * 80)
        if self.save_metrics_flag:
            tampered = (
                "tampered_only" if dataset.tampered_only else "tampered_and_pristine"
            )
            timestamp = time.strftime("%Y%m%d_%H:%M")

            report_id = f"{tampered}_{timestamp}"
            if self.save_heatmap:
                log.info("     - Saving heatmap metrics")
                self.save_metrics(output_path, heatmap_metrics, report_id)
            else:
                log.info("     - No heatmap metrics to save")
            if self.save_mask:
                log.info("     - Saving mask metrics")
                self.save_metrics(output_path, mask_metrics, report_id)
            else:
                log.info("     - No mask metrics to save")
            if self.save_detection:
                log.info("     - Saving detection metrics")
                self.save_metrics(output_path, detection_metrics, report_id)
            else:
                log.info("     - No detection metrics to save")
        else:
            log.info("     - Not saving metrics")
        log.info("-" * 80)
        log.info("Benchmark finished")
        log.info("-" * 80)
        log.info("-" * 80)

    def move_to_device(self, data):
        return {
            key: value.to(self.device, dtype=torch.float32)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in data.items()
        }

    def save_metrics(self, output_path, metrics, report_id):
        metrics_path = os.path.join(output_path, "metrics")
        os.makedirs(metrics_path, exist_ok=True)

        metric_report = metrics.compute()
        torch.save(
            metrics.state_dict(),
            os.path.join(metrics_path, f"{metrics.prefix}_state_{report_id}.pt"),
        )

        json_report = {}
        for key, value in metric_report.items():
            if isinstance(value, torch.Tensor) and value.dim() == 0:
                json_report[key] = float(value)
            elif isinstance(value, tuple) and all(
                isinstance(v, torch.Tensor) for v in value
            ):
                json_report[key] = [v.tolist() for v in value]
            elif (
                isinstance(value, int)
                or isinstance(value, float)
                or isinstance(value, str)
            ):
                json_report[key] = value
            else:
                log.warning(f"Skipping metric '{key}' of type '{type(value)}'")

        with open(
            os.path.join(metrics_path, f"{metrics.prefix}_report_{report_id}.json"), "w"
        ) as f:
            json.dump(json_report, f)

    def save_pred_output(self, output_path, image_name, output):
        image_save_path = os.path.join(output_path, "outputs", image_name)
        if not self.use_existing_output or not os.path.exists(
            os.path.join(image_save_path, "arrays.npz")
        ):
            os.makedirs(image_save_path, exist_ok=True)

            array_like_dict = {}
            non_array_like_dict = {}

            for key, value in output.items():
                if isinstance(value, (torch.Tensor)):
                    array_like_dict[key] = value.cpu()
                elif isinstance(value, (list, np.ndarray)):
                    array_like_dict[key] = value
                else:
                    non_array_like_dict[key] = value

            np.savez_compressed(
                os.path.join(image_save_path, "arrays"), **array_like_dict
            )
            if non_array_like_dict:
                with open(os.path.join(image_save_path, "data.json"), "w") as f:
                    json.dump(non_array_like_dict, f)
        else:
            log.io_message(
                f"Output for image '{image_name}' already exists. "
                f"Skipping saving output."
            )

    def check_for_existing_output(self, output_path, image_name):
        output_path = os.path.join(output_path, "outputs")
        if not os.path.exists(output_path):
            return None
        files = os.listdir(output_path)
        if image_name in files:
            log.io_message(
                f"Output for image '{image_name}' already exists. "
                f"Loading existing output."
            )

            prior_output = np.load(
                os.path.join(output_path, image_name, "arrays.npz"),
                allow_pickle=True,
            )

            prior_output = dict(prior_output)
            prior_output = {
                k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                for k, v in prior_output.items()
            }

            data_json_path = os.path.join(output_path, image_name, "data.json")
            if os.path.exists(data_json_path):
                with open(data_json_path, "r") as f:
                    data_json = json.load(f)
                prior_output.update(data_json)

            return prior_output

        log.io_message(f"No prior output found for image '{image_name}'.")
        return None
