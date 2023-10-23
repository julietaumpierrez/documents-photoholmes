import torch
from torchmetrics import JaccardIndex

from photoholmes.metrics.base import PhotoHolmesMetric


class IoU(PhotoHolmesMetric):
    """
    Intersection over Union (IoU) metric for image masks using torchmetrics as a
    wrapper.
    """

    def __init__(self):
        super().__init__()
        self.iou = JaccardIndex(task="binary")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the metric values based on the current batch of mask predictions and
        targets.

        Args:
            preds (torch.Tensor): Predicted masks.
            target (torch.Tensor): Ground truth masks.
        """
        self.iou.update(preds, target)

    def compute(self) -> torch.Tensor:
        """
        Compute the IoU based on the accumulated values.
        """
        return self.iou.compute()

    def reset(self) -> None:
        """
        Reset the metric values.
        """
        self.iou.reset()
