import torch
from torch import Tensor
from torchmetrics import Metric


class IoU_weighted_metric(Metric):
    """
    The IoU weighted (Intersection over Union weighted) metric calculates the IoU taking
    into account the value of the heatmap as a probability and uses weighted true
    positives, weighted false positives, weighted true negatives and weighted false
    negatives to calculate the IoU.

    Attributes:
        IoU weighted (torch.Tensor): A tensor that accumulates the count of IoU weighted
                                        across batches.

    Methods:
        __init__(**kwargs): Initializes the IoU weighted metric object.
        update(preds: Tensor, target: Tensor): Updates the states with a new batch of
                                               predictions and targets.
        compute() -> Tensor: Computes the IoU weighted over all batches.

    Example:
        >>> IoU_weighted_metric = IoU_weighted()
        >>> for preds_batch, targets_batch in data_loader:
        >>>     IoU_weighted_metric.update(preds_batch, targets_batch)
        >>> IoU_weighted = IoU_weighted_metric.compute()
    """

    def __init__(self, **kwargs):
        """
        Initializes the IoU weighted metric object.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.add_state("IoU_weighted", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_images", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Updates the IoU weighted counts with a new batch of
        predictions and targets. It assumes both predictions as heatmap or binary
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

        pred_flat = preds.flatten()
        target_flat = target.flatten()
        TPw = torch.sum(pred_flat * target_flat)
        FPw = torch.sum((1 - pred_flat) * target_flat)
        FNw = torch.sum(pred_flat * (1 - target_flat))
        denominator = TPw + FPw + FNw
        if denominator != 0:
            self.IoU_weighted += TPw / denominator
        self.total_images += torch.tensor(1)

    def compute(self) -> Tensor:
        """
        Computes the IoU weighted over all the batches averaging all the
        IoU wighted of each image.

        Returns:
            Tensor: The computed IoU weighted over the full dataset.
                    If the total number of images is zero,
                    it returns 0.0 to avoid division by zero.
        """
        IoU_weighted = self.IoU_weighted.float()
        total_images = self.total_images.float()
        return IoU_weighted / total_images if total_images != 0 else torch.tensor(0.0)
