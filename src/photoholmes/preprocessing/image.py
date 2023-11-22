from typing import Dict, Optional, TypeVar, Union

import numpy as np
import torch
from numpy.typing import NDArray
from PIL.Image import Image
from torch import Tensor

from photoholmes.preprocessing.base import PreprocessingTransform

T = TypeVar("T", Tensor, NDArray)


class Normalize(PreprocessingTransform):
    """
    Changes the image range from [0, 255] to [0, 1].

    Args:
        image: Image to be normalized.
        **kwargs: Additional keyword arguments to passthrough.
    Returns:
        A dictionary with the following key-value pairs:
            - "image": The normalized image.
            - **kwargs: The additional keyword arguments passed through unchanged.
    """

    def __call__(self, image: T, **kwargs) -> Dict[str, T]:
        if image.dtype == np.uint8 or image.dtype == torch.uint8:
            image = image / 255
        elif image.max() > 1:
            image = image / 255
        return {"image": image, **kwargs}


class ToTensor(PreprocessingTransform):
    """
    Converts a numpy array to a PyTorch tensor.

    Args:
        image: Image to be converted to a tensor.
        **kwargs: Additional keyword arguments to passthrough.

    Returns:
        A dictionary with the following key-value pairs:
            - "image": The input image as a PyTorch tensor.
            - **kwargs: The additional keyword arguments passed through unchanged.
    """

    def __call__(self, image: NDArray, **kwargs) -> Dict[str, Tensor]:
        t_image = torch.from_numpy(image)
        if t_image.ndim == 3:
            t_image = t_image.permute(2, 0, 1)

        for k in kwargs:
            if isinstance(kwargs[k], list):
                kwargs[k] = np.array(kwargs[k])
            kwargs[k] = torch.from_numpy(kwargs[k])

        return {"image": t_image, **kwargs}


class ToNumpy(PreprocessingTransform):
    """
    Converts inputs to numpy arrays. If input is already a numpy array,
    it leaves it as is.

    Args:
        image: Image to be converted to a tensor.
        **kwargs: Additional keyword arguments to passthrough.

    Returns:
        A dictionary with the following key-value pairs:
            - "image": The input image as a PyTorch tensor.
            - **kwargs: The additional keyword arguments passed through unchanged.
    """

    def __call__(
        self, image: Optional[Union[T, Image]] = None, **kwargs
    ) -> Dict[str, NDArray]:
        t_image = None
        if isinstance(image, Tensor):
            t_image = image.cpu().numpy()
        elif isinstance(image, np.ndarray):
            t_image = image.copy()
        elif image is not None:
            t_image = np.array(image)

        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                continue
            elif isinstance(v, Tensor):
                kwargs[k] = v.cpu().numpy()
            else:
                kwargs[k] = np.array(v)

        if t_image is None:
            return {**kwargs}
        else:
            return {"image": t_image, **kwargs}


class RGBtoGray(PreprocessingTransform):
    """
    Converts an RGB image to grayscale.

    Args:
        image: Image to be converted to grayscale.
        **kwargs: Additional keyword arguments to passthrough.

    Returns:
        A dictionary with the following key-value pairs:
            - "image": The input image as a grayscale numpy array or PyTorch tensor.
            - **kwargs: The additional keyword arguments passed through unchanged.
    """

    def __call__(self, image: T, **kwargs) -> Dict[str, T]:
        if isinstance(image, Tensor):
            image = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        else:
            image = (
                0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
            )
        return {"image": image, **kwargs}
