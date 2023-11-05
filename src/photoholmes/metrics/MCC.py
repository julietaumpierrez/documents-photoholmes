from torch import Tensor
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

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update the metric values based on the current batch of mask predictions and
        targets.

        Args:
            preds (Tensor): Predicted masks.
            target (Tensor): Ground truth masks.
        """
        self.mcc.update(preds, target)

    def compute(self) -> Tensor:
        """
        Compute the MCC based on the accumulated values.
        """
        return self.mcc.compute()

    def reset(self) -> None:
        """
        Reset the metric values.
        """
        self.mcc.reset()
