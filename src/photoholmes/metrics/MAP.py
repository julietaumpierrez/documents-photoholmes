from torch import Tensor
from torchmetrics.classification import AveragePrecision

from photoholmes.metrics.base import BaseMetric


class MAP(BaseMetric):
    """
    Mean Average Precision (MAP) metric for image masks using torchmetrics.
    """

    def __init__(self, permuted=False):
        super().__init__()
        self.ap = AveragePrecision(task="binary")
        self.permuted = permuted

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update the metric values based on the current batch of mask predictions and
        targets.

        Args:
            preds (Tensor): Predicted masks.
            target (Tensor): Ground truth masks.
        """
        self.ap.update(preds, target)

    def compute(self) -> Tensor:
        """
        Compute the Mean Average Precision (MAP) or Permuted MAP based on the
        accumulated values.

        Returns:
            Tensor: The computed MAP or Permuted MAP value.
        """
        map_value = self.ap.compute()

        if self.permuted:
            permuted_map_value = 1 - map_value
            return permuted_map_value
        else:
            return map_value

    def reset(self) -> None:
        """
        Reset the metric values.
        """
        self.ap.reset()
