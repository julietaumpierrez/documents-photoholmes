from typing import Dict

import numpy as np
import torch
from torch import Tensor

from photoholmes.methods.base import BaseMethod


class Naive(BaseMethod):
    """A random method to test the program structure"""

    def __init__(self, sigma: float = 1, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma

    def predict(self, image: np.ndarray, **kwargs) -> Dict[str, Tensor]:
        """Predicts masks from a list of images."""
        shape = image.shape[:2] if image.ndim > 2 else image.shape
        output = np.random.normal(0.5, self.sigma, size=shape)
        output = torch.from_numpy(output).float()
        return {"heatmap": output}
