from typing import List, Union

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.classification.auroc import binary_auroc


class mAuroc(Metric):
    """
    Compute the mean Area Under the Receiver Operating Characteristic curve (mAuroc).
    """

    def __init__(self, thresholds: Union[int, List[float], None] = None, **kwargs):
        super().__init__(**kwargs)

        self.thresholds = thresholds
        self.add_state("auroc", default=torch.tensor(0.0))
        self.add_state("total", default=torch.tensor(0.0))

    def update(self, preds: Tensor, target: Tensor):
        """ """
        bauroc = binary_auroc(preds, target, thresholds=self.thresholds)
        if bauroc == 0:
            return  # target is all 0 or all 1, not useful for mAuroc
        self.auroc += bauroc
        self.total += 1

    def compute(self):
        return self.auroc / self.total
