from typing import List, Union

from torch import Tensor
from torchmetrics import AUROC as auroc

from photoholmes.metrics.base import BaseMetric


class AUROC(BaseMetric):
    """
    AUROC metric for image masks using torchmetrics as a wrapper.
    """

    def __init__(self, thresholds: Union[int, List[float], Tensor, None] = None):
        super().__init__()
        self.auroc = auroc(task="binary", thresholds=thresholds)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update the metric values based on the current batch of mask predictions and
        targets.

        Args:
            preds (Tensor): Predicted masks.
            target (Tensor): Ground truth masks.
        """
        self.auroc.update(preds, target)

    def compute(self) -> Tensor:
        """
        Compute the ROC based on the accumulated values.
        """
        return self.auroc.compute()

    def reset(self) -> None:
        """
        Reset the metric values.
        """
        self.auroc.reset()

        self._update_count = 0
        self._forward_cache = None
        self._computed = None

        for attr, default in self._defaults.items():
            current_val = getattr(self, attr)
            if isinstance(default, Tensor):
                setattr(self, attr, default.detach().clone().to(current_val.device))
            else:
                setattr(self, attr, [])

        # reset internal states
        self._cache = None
        self._is_synced = False
