# %%
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import torch
from torchmetrics import Metric

# %%


class PhotoHolmesMetric(Metric, ABC):
    """
    Abstract class for metrics specific to image masks.
    """

    @abstractmethod
    def compute(self) -> torch.Tensor:
        """
        Compute the metric based on the accumulated values.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the metric values.
        """
        pass

    @abstractmethod
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the metric values based on the current batch of mask predictions and targets.

        Args:
            preds (torch.Tensor): Predicted masks.
            target (torch.Tensor): Ground truth masks.
        """
        pass


# %%


# %%
from torchmetrics import Recall


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


# %%
tpr = TPR()

# %%
from torchmetrics import Specificity


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


# %%
fpr = FPR()
# %%

# %%

from torchmetrics import Precision as precision


class Precision(PhotoHolmesMetric):
    """
    Precision metric for image masks using torchmetrics as a wrapper.
    """

    def __init__(self):
        super().__init__()
        self.precision = precision(task="binary")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the metric values based on the current batch of mask predictions and targets.

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


# %%
precision_ = Precision()
# %%

# %%

from torchmetrics import MatthewsCorrCoef


class MCC(PhotoHolmesMetric):
    """
    Matthews Correlation Coefficient (MCC) metric for image masks using torchmetrics as a wrapper.
    """

    def __init__(self):
        super().__init__()
        self.mcc = MatthewsCorrCoef(task="binary")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the metric values based on the current batch of mask predictions and targets.

        Args:
            preds (torch.Tensor): Predicted masks.
            target (torch.Tensor): Ground truth masks.
        """
        self.mcc.update(preds, target)

    def compute(self) -> torch.Tensor:
        """
        Compute the MCC based on the accumulated values.
        """
        return self.mcc.compute()

    def reset(self) -> None:
        """
        Reset the metric values.
        """
        self.mcc.reset()


# %%
mcc = MCC()

# %%
from torchmetrics import JaccardIndex


class IoU(PhotoHolmesMetric):
    """
    Intersection over Union (IoU) metric for image masks using torchmetrics as a wrapper.
    """

    def __init__(self):
        super().__init__()
        self.iou = JaccardIndex(task="binary")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the metric values based on the current batch of mask predictions and targets.

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


# %%

iou = IoU()

# %%
from torchmetrics import AUROC as auroc


class AUROC(PhotoHolmesMetric):
    """
    AUROC metric for image masks using torchmetrics as a wrapper.
    """

    def __init__(self, thresholds: Union[int, List[float], torch.Tensor, None] = None):
        super().__init__()
        self.auroc = auroc(task="binary", thresholds=thresholds)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the metric values based on the current batch of mask predictions and
        targets.

        Args:
            preds (torch.Tensor): Predicted masks.
            target (torch.Tensor): Ground truth masks.
        """
        self.auroc.update(preds, target)

    def compute(self) -> torch.Tensor:
        """
        Compute the ROC based on the accumulated values.
        """
        return self.auroc.compute()

    def reset(self) -> None:
        """
        Reset the metric values.
        """
        self.auroc.reset()


# %%
targets = torch.tensor([[[1, 1], [0, 0]], [[0, 0], [1, 1]]])
preds = torch.tensor([[[0.9, 0.48], [0.49, 0]], [[0, 0], [1, 0]]])
# %%
auroc_ = AUROC()
auroc_(preds, targets)

# %%
from torchmetrics import ROC as roc


class ROC(PhotoHolmesMetric):
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
        fpr, tpr, _ = self.compute()

        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.show()


# %%
targets = torch.tensor([[[1, 1], [0, 0]], [[0, 0], [1, 1]]])
preds = torch.tensor([[[0.9, 0.48], [0.49, 0]], [[0, 0], [1, 0]]])
# %%
roc_ = ROC()
roc_(preds, targets)
roc_.plot_roc_curve()
# %%
from torchmetrics.classification import AveragePrecision


class MAP(PhotoHolmesMetric):
    """
    Mean Average Precision (MAP) metric for image masks using torchmetrics.
    """

    def __init__(self, permuted=False):
        super().__init__()
        self.ap = AveragePrecision(task="binary")
        self.permuted = permuted

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the metric values based on the current batch of mask predictions and targets.

        Args:
            preds (torch.Tensor): Predicted masks.
            target (torch.Tensor): Ground truth masks.
        """
        self.ap.update(preds, target)

    def compute(self) -> torch.Tensor:
        """
        Compute the Mean Average Precision (MAP) or Permuted MAP based on the accumulated values.

        Returns:
            torch.Tensor: The computed MAP or Permuted MAP value.
        """
        map_value = self.ap.compute()

        if self.permuted:
            # Calculate the permuted MAP by subtracting the MAP from 1
            permuted_map_value = 1 - map_value
            return permuted_map_value
        else:
            return map_value

    def reset(self) -> None:
        """
        Reset the metric values.
        """
        self.ap.reset()


# %%
targets = torch.tensor([[[1, 1], [0, 0]], [[0, 0], [1, 1]]])
preds_binary = torch.tensor([[[0, 0], [0, 0]], [[0, 0], [1, 1]]], dtype=torch.float32)

# %%
map_ = MAP()
map_.update(preds_binary, targets)
# %%


class PRO(PhotoHolmesMetric):
    """
    Per-Region Overlap (PRO) Score metric for image masks.
    """

    def __init__(self):
        super().__init__()
        self.pred_regions = []
        self.target_regions = []

    def update(self, pred_mask: torch.Tensor, target_mask: torch.Tensor) -> None:
        """
        Update the metric values based on the current batch of predicted and target masks.

        Args:
            pred_mask (torch.Tensor): Predicted masks.
            target_mask (torch.Tensor): Ground truth masks.
        """
        self.pred_regions.append(pred_mask)
        self.target_regions.append(target_mask)

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the Per-Region Overlap (PRO) Score based on the accumulated values.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the PRO Score and the total number of regions.
        """
        # Combine all predicted and target regions into tensors
        pred_regions_tensor = torch.cat(self.pred_regions, dim=0)
        target_regions_tensor = torch.cat(self.target_regions, dim=0)

        # Calculate the PRO Score and the total number of regions
        pro_score = self.calculate_pro_score(pred_regions_tensor, target_regions_tensor)
        num_regions = (
            target_regions_tensor.sum()
        )  # Total number of regions in the ground truth

        return pro_score, num_regions

    def calculate_pro_score(
        self, pred_regions: torch.Tensor, target_regions: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the Per-Region Overlap (PRO) Score.

        Args:
            pred_regions (torch.Tensor): Predicted region masks.
            target_regions (torch.Tensor): Ground truth region masks.

        Returns:
            torch.Tensor: The computed PRO Score.
        """
        # Calculate the intersection and union of predicted and target regions
        intersection = (pred_regions * target_regions).sum()
        union = (pred_regions + target_regions).sum()

        # Calculate the PRO Score
        pro_score = intersection / union

        return pro_score

    def reset(self) -> None:
        """
        Reset the metric values.
        """
        self.pred_regions = []
        self.target_regions = []


# %%
