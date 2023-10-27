from typing import Dict, TypeVar

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from photoholmes.utils.preprocessing.base import BaseTransform

T = TypeVar("T", Tensor, NDArray)


class Normalize(BaseTransform):
    def __call__(self, image: T, **kwargs) -> Dict[str, T]:
        if image.dtype == np.uint8 or image.dtype == torch.uint8:
            image = image / 255
        elif image.max() > 1:
            image = image / 255
        return {"image": image, **kwargs}


class ToTensor(BaseTransform):
    def __call__(self, image: NDArray, **kwargs) -> Dict[str, Tensor]:
        t_image = torch.from_numpy(image)
        if t_image.ndim == 3:
            t_image = t_image.permute(2, 0, 1)

        for k in kwargs:
            if isinstance(kwargs[k], list):
                kwargs[k] = np.array(kwargs[k])
            kwargs[k] = torch.from_numpy(kwargs[k])

        return {"image": t_image, **kwargs}


class RGBtoGray(BaseTransform):
    def __call__(self, image: T, **kwargs) -> Dict[str, T]:
        if isinstance(image, Tensor):
            image = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        else:
            image = (
                0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
            )
        return {"image": image, **kwargs}
