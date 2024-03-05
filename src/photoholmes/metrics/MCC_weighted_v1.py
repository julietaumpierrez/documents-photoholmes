import torch
from torch import Tensor
from torchmetrics import Metric


class MCC_weighted_v1(Metric):
    """
    The MCC weighted (Mathews Correlation Coefficient weighted) metric calculates the
    MCC score taking into account the value of the heatmap as a probability and uses
    weighted true positives, weighted false positives, weighted true negatives and
    weighted false negatives to calculate the MCC score.
    This class computes de mean weighted MCC over the whole dataset. It calculates
    the weighted MCC score for each image and then averages to output a single result.

    Attributes:
        MCC score weighted (torch.Tensor): A tensor that accumulates the count of MCC
                                                score weighted across batches.

    Methods:
        __init__(**kwargs): Initializes the MCC score weighted metric object.
        update(preds: Tensor, target: Tensor): Updates the states with a new pair of
                                               prediction and target.
        compute() -> Tensor: Computes the MCC score weighted over all batches.

    Example:
        >>> MCC_weighted_metric = MCC_weighted()
        >>> for preds_batch, targets_batch in data_loader:
        >>>     MCC_weighted_metric.update(preds_batch, targets_batch)
        >>> mcc_weighted = MCC_weighted_metric.compute()
    """

    def __init__(self, **kwargs):
        """
        Initializes the MCC score weighted metric object.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.add_state("MCC_weighted", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_images", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Updates the MCC score weighted counts with a new pair of
        prediction and target. It assumes both predictions as heatmap or binary
        and binary targets.

        Args:
            preds (Tensor): The predictions from the model.
                Expected to be a binary tensor or a heatmap.
            target (Tensor): The ground truth labels. Expected to be a binary tensor.

        Raises:
            ValueError: If the shapes of predictions and targets do not match.
        """
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        target = target.to(torch.int)
        pred_flat = preds.flatten()
        target_flat = target.flatten()
        TPw = torch.sum(pred_flat * target_flat)
        FPw = torch.sum((1 - pred_flat) * target_flat)
        TNw = torch.sum((1 - pred_flat) * (1 - target_flat))
        FNw = torch.sum(pred_flat * (1 - target_flat))
        if TPw + TNw != 0 and FPw + FNw == 0:
            self.MCC_weighted += 1.0
        if TPw + TNw == 0 and FPw + FNw != 0:
            self.MCC_weighted += -1.0
        denominator = torch.sqrt((TPw + FPw) * (TPw + FNw) * (TNw + FPw) * (TNw + FNw))
        if denominator != 0:
            self.MCC_weighted += (TPw * TNw - FPw * FNw) / denominator

        self.total_images += torch.tensor(1)

    def compute(self) -> Tensor:
        """
        Computes the MCC weighted over all the batches averaging all the
        MCC wighted of each image.

        Returns:
            Tensor: The computed MCC weighted over the full dataset.
                    If the total number of images is zero,
                    it returns 0.0 to avoid division by zero.
        """
        mcc_weighted = self.MCC_weighted.float()
        total_images = self.total_images.float()
        return mcc_weighted / total_images if total_images != 0 else torch.tensor(0.0)
