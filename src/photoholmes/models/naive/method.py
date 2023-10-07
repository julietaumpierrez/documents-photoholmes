from typing import Any, Dict, Optional

import numpy as np

from photoholmes.models.base import BaseMethod
from photoholmes.utils.generic import load_yaml


class Naive(BaseMethod):
    """A random method to test the program structure"""

    def __init__(self, sigma: float = 1, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predicts masks from a list of images."""
        shape = image.shape[:2] if image.ndim > 2 else image.shape
        return np.random.normal(0.5, self.sigma, size=shape)
