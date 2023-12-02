from torch import Tensor
from torchmetrics import Recall

from photoholmes.metrics.base import BaseMetric


class TPR(BaseMetric):
    """
    TPR (True Positive Rate) metric for image masks using torchmetrics as a wrapper.
    """

    def __init__(self):
        super().__init__()
        self.recall = Recall(task="binary")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update the metric values based on the current batch of mask predictions and targets.

        Args:
            preds (Tensor): Predicted masks.
            target (Tensor): Ground truth masks.
        """
        self.recall.update(preds, target)

    def compute(self) -> Tensor:
        """
        Compute the TPR based on the accumulated values.
        """
        return self.recall.compute()

    def reset(self) -> None:
        """
        Reset the metric values.
        """
        self.recall.reset()
