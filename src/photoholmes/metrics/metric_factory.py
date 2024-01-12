from typing import List, Union

from torchmetrics import Metric

from photoholmes.metrics.registry import MetricName


class MetricFactory:
    """
    MetricFactory class responsible for creating metric instances.

    Supported Metrics:
        - AUROC (Area Under the Receiver Operating Characteristic curve)
        - FPR (False Positive Rate)
        - IoU (Intersection over Union, also known as Jaccard Index)
        - MCC (Matthews Correlation Coefficient)
        - Precision
        - ROC (Receiver Operating Characteristic curve)
        - TPR (True Positive Rate, synonymous with Recall)

    Methods:
        load(metric_names: List[Union[str, MetricName]]) -> List[Metric]:
            Instantiates and returns a list of metric objects corresponding to the
            specified metric names.
    """

    @staticmethod
    def load(metric_names: List[Union[str, MetricName]]) -> List[Metric]:
        """
        Instantiates and returns a list of metric objects corresponding to the specified
        metric names.

        Args:
            metric_names (List[Union[str, MetricName]]): A list of the names of the
                metrics to load.
                These can be strings representing the metric names or instances of the
                MetricName enum.

        Returns:
            List[Metric]: A list of metric objects corresponding to the provided metric
                names.
                The order of the metric objects in the list will correspond to the
                order of names provided.

        Raises:
            ValueError: If the 'metric_names' list is empty, indicating that no metric
                names have been specified.
            NotImplementedError: If any of the metric names provided are not recognized
                or not implemented in the PhotoHolmes library.

        Examples:
            Loading a single metric:
            >>> metrics = MetricFactory.load(["auroc"])

            Loading multiple metrics:
            >>> metrics = MetricFactory.load(["auroc", MetricName.PRECISION])

        """
        if not metric_names:
            raise ValueError("metric_names cannot be empty.")
        metrics = []
        for metric_name in metric_names:
            if isinstance(metric_name, str):
                metric_name = MetricName(metric_name.lower())
            # TODO: Add mAP metric
            match metric_name:
                case MetricName.AUROC:
                    from torchmetrics import AUROC

                    metrics.append(AUROC(task="binary"))
                case MetricName.FPR:
                    from photoholmes.metrics.FPR import FPR

                    metrics.append(FPR())
                case MetricName.IoU:
                    from torchmetrics import JaccardIndex as IoU

                    metrics.append(IoU(task="binary"))
                case MetricName.MCC:
                    from torchmetrics import MatthewsCorrCoef

                    metrics.append(MatthewsCorrCoef(task="binary"))
                case MetricName.Precision:
                    from torchmetrics import Precision

                    metrics.append(Precision(task="binary"))
                case MetricName.ROC:
                    from torchmetrics import ROC

                    metrics.append(ROC(task="binary"))
                case MetricName.TPR:
                    from torchmetrics import Recall as TPR

                    metrics.append(TPR(task="binary"))
                case MetricName.IoU_WEIGHTED:
                    from photoholmes.metrics.IoU_weighted import IoU_weighted

                    metrics.append(IoU_weighted())
                case MetricName.F1_WEIGHTED:
                    from photoholmes.metrics.F1_weighted import F1_weighted

                    metrics.append(F1_weighted())
                case MetricName.MCC_WEIGHTED:
                    from photoholmes.metrics.MCC_weighted import MCC_weighted

                    metrics.append(MCC_weighted())
                case _:
                    raise NotImplementedError(
                        f"Metric '{metric_name}' is not implemented."
                    )
        return metrics
