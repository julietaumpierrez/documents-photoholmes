import torch
from torchmetrics import Recall

from photoholmes.metrics.base import PhotoHolmesMetric


class TPR(PhotoHolmesMetric):
    """
    TPR (True Positive Rate) metric for image masks using torchmetrics as a wrapper.
    """

    def __init__(self):
        super().__init__()
        self.recall = Recall(task="binary")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the metric values based on the current batch of mask predictions and targets.

        Args:
            preds (torch.Tensor): Predicted masks.
            target (torch.Tensor): Ground truth masks.
        """
        self.recall.update(preds, target)

    def compute(self) -> torch.Tensor:
        """
        Compute the TPR based on the accumulated values.
        """
        return self.recall.compute()

    def reset(self) -> None:
        """
        Reset the metric values.
        """
        self.recall.reset()
