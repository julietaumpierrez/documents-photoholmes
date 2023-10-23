from typing import List, Union

import torch
from torchmetrics import AUROC as auroc

from photoholmes.metrics.base import PhotoHolmesMetric


class AUROC(PhotoHolmesMetric):
    """
    AUROC metric for image masks using torchmetrics as a wrapper.
    """

    def __init__(self, thresholds: Union[int, List[float], torch.Tensor, None] = None):
        super().__init__()
        self.auroc = auroc(task="binary", thresholds=thresholds)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the metric values based on the current batch of mask predictions and
        targets.

        Args:
            preds (torch.Tensor): Predicted masks.
            target (torch.Tensor): Ground truth masks.
        """
        self.auroc.update(preds, target)

    def compute(self) -> torch.Tensor:
        """
        Compute the ROC based on the accumulated values.
        """
        return self.auroc.compute()

    def reset(self) -> None:
        """
        Reset the metric values.
        """
        self.auroc.reset()
