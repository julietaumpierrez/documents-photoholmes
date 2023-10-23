from enum import Enum, unique

from .AUROC import AUROC
from .FPR import FPR
from .IoU import IoU
from .MAP import MAP
from .MCC import MCC
from .Precision import Precision
from .ROC import ROC
from .TPR import TPR


@unique
class MetricType(Enum):
    AUROC = 0
    FPR = 1
    IoU = 2
    MAP = 3
    MCC = 4
    Precision = 5
    ROC = 6
    TPR = 7


def string_to_metric_type(metric_name: str) -> MetricType:
    if metric_name == "auroc":
        return MetricType.AUROC
    elif metric_name == "fpr":
        return MetricType.FPR
    elif metric_name == "iou":
        return MetricType.IoU
    elif metric_name == "map":
        return MetricType.MAP
    elif metric_name == "mcc":
        return MetricType.MCC
    elif metric_name == "precision":
        return MetricType.Precision
    elif metric_name == "roc":
        return MetricType.ROC
    elif metric_name == "tpr":
        return MetricType.TPR
    else:
        raise Exception("Metric Type not implemented yet.")


class MetricFactory:
    @staticmethod
    def create(metric_name: str):
        """Instantiates a metric corresponding to the name passed."""
        metric_type = string_to_metric_type(metric_name)
        if metric_type == MetricType.AUROC:
            return AUROC()
        elif metric_type == MetricType.FPR:
            return FPR()
        elif metric_type == MetricType.IoU:
            return IoU()
        elif metric_type == MetricType.MAP:
            return MAP()
        elif metric_type == MetricType.MCC:
            return MCC()
        elif metric_type == MetricType.Precision:
            return Precision()
        elif metric_type == MetricType.ROC:
            return ROC()
        elif metric_type == MetricType.TPR:
            return TPR()
        else:
            raise Exception("Selected metric_name is not defined")
