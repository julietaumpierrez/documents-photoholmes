from abc import ABC, abstractmethod
from typing import Any

import torch
from torchmetrics import Metric


class PhotoHolmesMetric(Metric, ABC):
    """
    Abstract class for metrics specific to image masks.
    """

    @abstractmethod
    def compute(self) -> torch.Tensor:
        """
        Compute the metric based on the accumulated values.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the metric values.
        """
        pass

    @abstractmethod
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the metric values based on the current batch of mask predictions and targets.

        Args:
            preds (torch.Tensor): Predicted masks.
            target (torch.Tensor): Ground truth masks.
        """
        pass
