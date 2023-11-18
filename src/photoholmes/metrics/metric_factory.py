from typing import Union

from photoholmes.metrics.base import BaseMetric
from photoholmes.metrics.registry import MetricName


class MetricFactory:
    @staticmethod
    def load(metric_name: Union[str, MetricName]) -> BaseMetric:
        """Instantiates a metric corresponding to the name passed."""
        if isinstance(metric_name, str):
            metric_name = MetricName(metric_name.lower())

        match metric_name:
            case MetricName.AUROC:
                from photoholmes.metrics.AUROC import AUROC

                return AUROC()
            case MetricName.FPR:
                from photoholmes.metrics.FPR import FPR

                return FPR()
            case MetricName.IoU:
                from photoholmes.metrics.IoU import IoU

                return IoU()
            case MetricName.MAP:
                from photoholmes.metrics.MAP import MAP

                return MAP()
            case MetricName.MCC:
                from photoholmes.metrics.MCC import MCC

                return MCC()
            case MetricName.Precision:
                from photoholmes.metrics.Precision import Precision

                return Precision()
            case MetricName.ROC:
                from photoholmes.metrics.ROC import ROC

                return ROC()
            case MetricName.TPR:
                from photoholmes.metrics.TPR import TPR

                return TPR()
            case _:
                raise NotImplementedError(f"Metric '{metric_name}' is not implemented.")
