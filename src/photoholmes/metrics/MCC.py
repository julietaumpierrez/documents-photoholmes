import torch
from torchmetrics import MatthewsCorrCoef

from photoholmes.metrics.base import BaseMetric


class MCC(BaseMetric):
    """
    Matthews Correlation Coefficient (MCC) metric for image masks using torchmetrics as
    a wrapper.
    """

    def __init__(self):
        super().__init__()
        self.mcc = MatthewsCorrCoef(task="binary")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the metric values based on the current batch of mask predictions and
        targets.

        Args:
            preds (torch.Tensor): Predicted masks.
            target (torch.Tensor): Ground truth masks.
        """
        self.mcc.update(preds, target)

    def compute(self) -> torch.Tensor:
        """
        Compute the MCC based on the accumulated values.
        """
        return self.mcc.compute()

    def reset(self) -> None:
        """
        Reset the metric values.
        """
        self.mcc.reset()
