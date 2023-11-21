from typing import Union

from photoholmes.metrics.base import BaseMetric
from photoholmes.metrics.registry import MetricName


class MetricFactory:
    @staticmethod
    def load(metric_name: Union[str, MetricName]) -> BaseMetric:
        """
        Instantiates and returns a metric object corresponding to the specified
        metric name.

        Args:
            metric_name (Union[str, MetricName]): The name of the metric to load.
                Can be a string or a MetricName enum instance.

        Returns:
            BaseMetric: An instance of a subclass of photoholmes.metrics.base.BaseMetric
                corresponding to the provided metric name.

        Raises:
            NotImplementedError: If the metric name provided is not recognized or not
                implemented.

        Examples:
            >>> metric = MetricFactory.load("auroc")
            >>> metric = MetricFactory.load(MetricName.PRECISION)
        """
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
