from torch import Tensor
from torchmetrics import Precision as precision

from photoholmes.metrics.base import BaseMetric


class Precision(BaseMetric):
    """
    Precision metric for image masks using torchmetrics as a wrapper.
    """

    def __init__(self):
        super().__init__()
        self.precision = precision(task="binary")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update the metric values based on the current batch of mask predictions and
        targets.

        Args:
            preds (Tensor): Predicted masks.
            target (Tensor): Ground truth masks.
        """
        self.precision.update(preds, target)

    def compute(self) -> Tensor:
        """
        Compute the Precision based on the accumulated values.
        """
        return self.precision.compute()

    def reset(self) -> None:
        """
        Reset the metric values.
        """
        self.precision.reset()
