import torch
from torchmetrics import Specificity

from photoholmes.metrics.base import PhotoHolmesMetric


class FPR(PhotoHolmesMetric):
    """
    FPR (False Positive Rate) metric for image masks using torchmetrics as a wrapper.
    """

    def __init__(self):
        super().__init__()
        self.specificity = Specificity(task="binary")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the metric values based on the current batch of mask predictions and targets.

        Args:
            preds (torch.Tensor): Predicted masks.
            target (torch.Tensor): Ground truth masks.
        """
        self.specificity.update(preds, target)

    def compute(self) -> torch.Tensor:
        """
        Compute the FPR based on the accumulated values.
        """
        tnr = self.specificity.compute()
        return 1 - tnr

    def reset(self) -> None:
        """
        Reset the metric values.
        """
        self.specificity.reset()
