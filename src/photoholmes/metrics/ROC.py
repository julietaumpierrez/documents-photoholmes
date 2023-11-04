from typing import Tuple

import matplotlib.pyplot as plt
import torch
from torchmetrics import ROC as roc

from photoholmes.metrics.base import BaseMetric


class ROC(BaseMetric):
    """
    ROC metric for image masks using torchmetrics as a wrapper.
    """

    def __init__(self):
        super().__init__()
        self.roc = roc(task="binary")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the metric values based on the current batch of mask predictions and targets.

        Args:
            preds (torch.Tensor): Predicted masks.
            target (torch.Tensor): Ground truth masks.
        """
        self.roc.update(preds, target)

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the ROC based on the accumulated values.
        """
        return self.roc.compute()

    def reset(self) -> None:
        """
        Reset the metric values.
        """
        self.roc.reset()

    def plot_roc_curve(self) -> None:
        """
        Plot the ROC curve using accumulated values.
        """
        fpr, tpr, _ = self.roc.compute()

        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
