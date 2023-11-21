from abc import ABC, abstractmethod

from torch import Tensor
from torchmetrics import Metric


class BaseMetric(Metric, ABC):
    """
    Abstract class for metrics specific to image masks.
    """

    @abstractmethod
    def compute(self) -> Tensor:
        """
        Compute the metric based on the accumulated values.
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the metric values.
        """

    @abstractmethod
    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update the metric values based on the current batch of mask predictions and targets.

        Args:
            preds (Tensor): Predicted masks.
            target (Tensor): Ground truth masks.
        """
