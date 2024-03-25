import glob
import logging
import os
import shutil
from enum import Enum
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class NeededEvals(Enum):
    TPOnly = 0
    TPandPT = 1
    BOTH = 2


DATASET_TP_MAPPING = {
    "columbiadataset": NeededEvals.BOTH,
    "columbiawebpdataset": NeededEvals.BOTH,
    "columbiaosndataset": NeededEvals.TPOnly,
    "casia1copymovedataset": NeededEvals.BOTH,
    "casia1splicingdataset": NeededEvals.BOTH,
    "coveragedataset": NeededEvals.TPandPT,
    "dso1dataset": NeededEvals.BOTH,
    "tracenoiseexodataset": NeededEvals.TPOnly,
    "tracenoiseendodataset": NeededEvals.TPOnly,
    "tracecfaalgexodataset": NeededEvals.TPOnly,
    "tracecfaalgendodataset": NeededEvals.TPOnly,
    "tracecfagridendodataset": NeededEvals.TPOnly,
    "tracecfagridexodataset": NeededEvals.TPOnly,
    "tracehybridendodataset": NeededEvals.TPOnly,
    "tracehybridexodataset": NeededEvals.TPOnly,
    "tracejpeggridendodataset": NeededEvals.TPOnly,
    "tracejpeggridexodataset": NeededEvals.TPOnly,
    "tracejpegqualityexodataset": NeededEvals.TPOnly,
    "tracejpegqualityendodataset": NeededEvals.TPOnly,
    "dso1osndataset": NeededEvals.TPOnly,
    "realistictamperingdataset": NeededEvals.TPandPT,
    "realistictamperingwebpdataset": NeededEvals.TPandPT,
    "autosplice100dataset": NeededEvals.BOTH,
    "autosplice90dataset": NeededEvals.TPOnly,
    "autosplice75dataset": NeededEvals.TPOnly,
}


METHOD_OUTPUT_MAPPINGS = {
    "adaptive_cfa_net": ["heatmap"],
    "catnet": ["heatmap"],
    "dq": ["heatmap"],
    "exif_as_language": ["heatmap", "mask", "detection"],
    "noisesniffer": ["mask", "detection"],
    "psccnet": ["heatmap", "detection"],
    "zero": ["mask", "detection"],
    "trufor": ["heatmap", "detection"],
    "focal": ["heatmap"],
    "splicebuster": ["heatmap"],
}


def check_evals(dataset: str, evals: List[str], needed_evals: NeededEvals, output: str):
    lastest_runs = []
    if needed_evals == NeededEvals.BOTH or needed_evals == NeededEvals.TPOnly:
        tampered_only = [e for e in evals if "tampered_only" in e]
        if len(tampered_only) == 0:
            logger.warning(
                f"Missing tampered only evaluation for {output} in dataset {dataset}"
            )
        else:
            tampered_only.sort()
            lastest_runs.append(tampered_only[-1])
    if needed_evals == NeededEvals.BOTH or needed_evals == NeededEvals.TPandPT:
        complete_runs = [e for e in evals if "tampered_and_pristine" in e]
        if len(complete_runs) == 0:
            logger.warning(f"Missing full evaluation for {output} in dataset {dataset}")
        else:
            complete_runs.sort()
            lastest_runs.append(complete_runs[-1])
    return lastest_runs


def process_output_folder(method: str, output_dir: Path, upload_dir: Path):
    if method not in METHOD_OUTPUT_MAPPINGS:
        logger.error(f"Method {method} not found in method output mappings.")
        return

    datasets = os.listdir(output_dir / method)
    eval_datasets = DATASET_TP_MAPPING.copy()
    for d in datasets:
        if d not in eval_datasets:
            logger.warning(f"Dataset {d} isn't part of the evaluation.")
            continue

        dataset_dir = output_dir / method / d / "metrics"
        if not os.path.exists(dataset_dir):
            logger.warning(f"Dataset {d} is missing the metrics folder.")
            continue

        latest_reports = []
        needed_evals = eval_datasets.pop(d)
        for method_output in METHOD_OUTPUT_MAPPINGS[method]:
            reports = [
                str(f)
                for f in glob.glob(
                    f"*/{method_output}_report.json", root_dir=dataset_dir
                )
            ]
            latest_reports.extend(check_evals(d, reports, needed_evals, method_output))

        for report in latest_reports:
            report_dir = "/".join(report.split("/")[:-1])
            os.makedirs(os.path.join(upload_dir, method, d, report_dir), exist_ok=True)
            shutil.copy(
                os.path.join(dataset_dir, report),
                os.path.join(upload_dir, method, d, report_dir),
            )

    for d in eval_datasets.keys():
        logger.warning(f"Dataset {d} is missing from the output folder.")
