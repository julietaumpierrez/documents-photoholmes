from typing import Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from photoholmes.postprocessing.resizing import upscale_mask


def postprocessing_splicebuster(
    heatmap: NDArray, coords: Tuple[NDArray, NDArray], X: int, Y: int
) -> Tensor:
    """
    Postprocessing for splicebuster.

    Args:
        heatmap: splicebuster output
        coords: coordinates of the heatmap used to upscale it
        X: height of the image
        Y: length of the image
    Returns:
        heatmap:
    """

    heatmap = heatmap / np.max(heatmap)
    heatmap = upscale_mask(coords, heatmap, (X, Y), method="linear", fill_value=0)
    heatmap = torch.from_numpy(heatmap).float()

    return heatmap
