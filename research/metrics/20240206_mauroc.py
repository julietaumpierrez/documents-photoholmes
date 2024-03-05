# %%

# %%
from typing import List, Union

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.classification.auroc import binary_auroc


class mAuroc(Metric):
    """ """

    def __init__(self, thresholds: Union[int, List[float], None] = None, **kwargs):
        super().__init__(**kwargs)

        self.thresholds = thresholds
        self.add_state("auroc", default=torch.tensor(0.0))
        self.add_state("total", default=torch.tensor(0.0))

    def update(self, preds: Tensor, target: Tensor):
        """ """
        bauroc = binary_auroc(preds, target, thresholds=self.thresholds)
        self.auroc += bauroc
        self.total += 1

    def compute(self):
        return self.auroc / self.total


m = mAuroc()
# %%
m.update(torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0, 1, 0]))
m.update(torch.tensor([0.1, 0.2, 0]), torch.tensor([0, 1, 0]))
print(m.compute())

# %%
