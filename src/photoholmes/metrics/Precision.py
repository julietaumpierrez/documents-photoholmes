import torch
from torchmetrics import Precision as precision

from photoholmes.metrics.base import PhotoHolmesMetric


class Precision(PhotoHolmesMetric):
    """
    Precision metric for image masks using torchmetrics as a wrapper.
    """

    def __init__(self):
        super().__init__()
        self.precision = precision(task="binary")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the metric values based on the current batch of mask predictions and
        targets.

        Args:
            preds (torch.Tensor): Predicted masks.
            target (torch.Tensor): Ground truth masks.
        """
        self.precision.update(preds, target)

    def compute(self) -> torch.Tensor:
        """
        Compute the Precision based on the accumulated values.
        """
        return self.precision.compute()

    def reset(self) -> None:
        """
        Reset the metric values.
        """
        self.precision.reset()
