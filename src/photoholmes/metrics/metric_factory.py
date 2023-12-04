from typing import Union

from torchmetrics import Metric

from photoholmes.metrics.registry import MetricName


class MetricFactory:
    """
    MetricFactory class responsible for creating metric instances.

    Methods:
        load(metric_name: Union[str, MetricName]) -> Metric:
            Instantiates and returns a metric object based on the specified metric name.

    Supported Metrics:
        - AUROC (Area Under the Receiver Operating Characteristic curve)
        - FPR (False Positive Rate)
        - IoU (Intersection over Union, also known as Jaccard Index)
        - MCC (Matthews Correlation Coefficient)
        - Precision
        - ROC (Receiver Operating Characteristic curve)
        - TPR (True Positive Rate, synonymous with Recall)

    Examples:
        To create an AUROC metric:
        >>> metric = MetricFactory.load("auroc")

        To create a Precision metric using the MetricName enum:
        >>> metric = MetricFactory.load(MetricName.PRECISION)
    """

    @staticmethod
    def load(metric_name: Union[str, MetricName]) -> Metric:
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
                from torchmetrics import AUROC

                return AUROC(task="binary")
            case MetricName.FPR:
                from photoholmes.metrics.FPR import FPR

                return FPR()
            case MetricName.IoU:
                from torchmetrics import JaccardIndex as IoU

                return IoU(task="binary")
            case MetricName.MCC:
                from torchmetrics import MatthewsCorrCoef

                return MatthewsCorrCoef(task="binary")
            case MetricName.Precision:
                from torchmetrics import Precision

                return Precision(task="binary")
            case MetricName.ROC:
                from torchmetrics import ROC

                return ROC(task="binary")
            case MetricName.TPR:
                from torchmetrics import Recall as TPR

                return TPR(task="binary")
            case _:
                raise NotImplementedError(f"Metric '{metric_name}' is not implemented.")
