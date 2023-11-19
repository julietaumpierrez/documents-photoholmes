# %%
import os

from cv2 import norm

if "research" in os.path.abspath("."):
    os.chdir("../../")

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")


# %%
from typing import List, TypeVar, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from photoholmes.utils.preprocessing.base import PreprocessingTransform

T = TypeVar("T", Tensor, NDArray)


class Normalize_PH(PreprocessingTransform):
    """
    Normalize an image.

    Args:
        image: Image to be normalized.
        **kwargs: Additional keyword arguments to passthrough.
    Returns:
        A dictionary with the following key-value pairs:
            - "image": The normalized image.
            - **kwargs: The additional keyword arguments passed through unchanged.
    """

    def __init__(self, mean: Union[List[int], T], std: Union[List[int], T]) -> None:
        if isinstance(mean, list):
            self.mean = np.array(mean)
        else:
            self.mean = mean
        if isinstance(std, list):
            self.std = np.array(std)
        else:
            self.std = std

    def __call__(self, image: T, **kwargs):
        if isinstance(image, Tensor):
            mean = torch.as_tensor(self.mean)
            std = torch.as_tensor(self.std)

            image = (image.float() - mean.view(3, 1, 1)) / std.view(3, 1, 1)

        else:
            mean = self.mean
            std = self.std

            image = (image.astype(np.float32) - mean.reshape((1, 1, -1))) / std.reshape(
                (1, 1, -1)
            )

        return {"image": image, **kwargs}


# %%
import time

# %%
normalize = Normalize_PH(mean=[1, 1, 1], std=[1, 1, 1])
image = torch.normal(mean=1, std=1, size=(3, 200, 200))

t0 = time.time_ns()
norm_image = normalize(image=image)["image"]
tf = time.time_ns()
print(f"PH: {tf - t0} ns")
# %%
from torchvision.transforms import Normalize as Normalize_torch

normalize_torch = Normalize_torch(mean=[1, 1, 1], std=[1, 1, 1])

t0 = time.time_ns()
norm_image_torch = normalize_torch(image)
tf = time.time_ns()
print(f"Torch: {tf - t0} ns")

# %%
image_numpy = image.permute(1, 2, 0).numpy()
norm_image_numpy = normalize(image_numpy)["image"]
# %%
norm_image_torch == torch.from_numpy(norm_image_numpy).permute(2, 0, 1)
# %%
