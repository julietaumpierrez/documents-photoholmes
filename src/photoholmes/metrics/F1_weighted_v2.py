import torch
from torch import Tensor
from torchmetrics import Metric


class F1_weighted_v2(Metric):
    """
    The F1 weighted (F1 score weighted) metric calculates the F1 score taking
    into account the value of the heatmap as a probability and uses weighted true
    positives, weighted false positives, weighted true negatives and weighted false
    negatives to calculate the F1 score.

    Attributes:
        TPw (torch.Tensor): A tensor that accumulates the count of weighted true positives.
        FNw (torch.Tensor): A tensor that accumulates the count of weighted false negatives.
        FPw (torch.Tensor): A tensor that accumulates the count of weighted false positives.

    Methods:
        __init__(**kwargs): Initializes the F1 score weighted metric object.
        update(preds: Tensor, target: Tensor): Updates the states with a new batch of
                                               predictions and targets.
        compute() -> Tensor: Computes the F1 score weighted over all batches.

    Example:
        >>> F1_weighted_metric = F1_weighted()
        >>> for preds_batch, targets_batch in data_loader:
        >>>     F1_weighted_metric.update(preds_batch, targets_batch)
        >>> f1_weighted = F1_weighted_metric.compute()
    """

    def __init__(self, **kwargs):
        """
        Initializes the F1 score weighted metric object.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.add_state("TPw", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("FNw", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("FPw", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_images", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Updates the F1 score weighted counts with a new batch of
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

        target = target.to(torch.int)
        pred_flat = preds.flatten()
        target_flat = target.flatten()
        TPw = torch.sum(pred_flat * target_flat)
        FPw = torch.sum((1 - pred_flat) * target_flat)
        FNw = torch.sum(pred_flat * (1 - target_flat))
        self.TPw += TPw
        self.FNw += FNw
        self.FPw += FPw
        self.total_images += torch.tensor(1)

    def compute(self) -> Tensor:
        """
        Computes the F1 weighted over all the batches averaging all the
        F1 weighted of each image.

        Returns:
            Tensor: The computed F1 weighted over the full dataset.
                    If the total number of true positives is zero,
                    it returns 0.0 to avoid division by zero.
        """
        if not self.total_images:
            return torch.tensor(0.0)
        TPw = self.TPw.float()
        FNw = self.FNw.float()
        FPw = self.FPw.float()
        denominator = 2 * TPw + FNw + FPw
        f1_weighted = 2 * TPw / denominator if denominator != 0 else torch.tensor(0.0)
        return f1_weighted
