from typing import Dict, Tuple

import numpy as np
from torch import Tensor

from photoholmes.methods.base import BaseMethod
from photoholmes.postprocessing.image import to_tensor_dict


class Naive(BaseMethod):
    """A random method to test the program structure"""

    def __init__(self, sigma: float = 1, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma

    def predict(
        self, original_image_size=Tuple[int, int], **kwargs
    ) -> Dict[str, Tensor]:
        """Predicts masks from a list of images."""
        output = np.random.normal(0.5, self.sigma, size=original_image_size)
        return to_tensor_dict({"heatmap": output})
